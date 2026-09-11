--[[ agent.lua -- the fly, with no Isaac API in sight.

  Holds the brain, the heading and the speed, and runs one sense -> think ->
  act cycle per call.  It takes a plain description of what is around the fly
  and returns a plain movement command, so exactly the same code path can be
  driven by the game (mod/main.lua) or by the offline arena
  (tools/behaviour.lua) that checks the fly actually behaves like a fly.

  percepts = {
    landmarkAz = <deg or nil>,          -- the player, relative to the heading
    threats  = { {az=, angSize=, closing=, weight=}, ... },
    objects  = { {az=, angSize=}, ... },
    walls    = { <az>, ... },           -- directions that are blocked
  }
  All azimuths are in degrees relative to the fly's current heading, positive
  to the fly's right.
]]

-- The Brain/Sensory/Motor modules are passed in by the host rather than
-- required here, because Isaac loads modules with include() and the offline
-- harness uses require().
local Agent = {}
Agent.__index = Agent

local cos, sin, exp, pi = math.cos, math.sin, math.exp, math.pi

Agent.defaults = {
  stepsPerFrame   = 33,     -- 1 ms per step: brain time == game time at 30 fps
  minSteps        = 8,

  baseSpeed       = 2.6,
  maxSpeed        = 9.0,
  turnGain        = 0.085,  -- radians per frame per unit of turn signal
  maxTurn         = 0.30,
  forwardGain     = 0.035,
  escapeThreshold = 22.0,   -- Hz, mean over the escape descending neurons
  escapeBoost     = 5.5,

  loomPeak        = 165.0,
  loomSigma       = 34.0,
  objectPeak      = 90.0,
  objectSigma     = 40.0,
  barPeak         = 55.0,   -- LC9/LC10: object position, reaches the premotor side
  barSigma        = 30.0,
  wallPeak        = 95.0,
  wallSigma       = 30.0,

  -- MeTu -> TuBu -> ER -> EPG, the sky-compass pathway.  Off by default.
  -- The pathway is real and measurably azimuth-selective as far as the ring
  -- neurons (pipeline/probe_compass.py), but the EPG ring attractor does not
  -- form a bump in this model, so downstream of ER it produces no behaviour --
  -- while costing about half the simulation's CPU.  Set landmarkPeak above 0
  -- to switch it on anyway (the `flybrain compass` console command does).
  landmarkPeak    = 0.0,
  landmarkSigma   = 30.0,
}

function Agent.new(BrainClass, Sensory, Motor, net, meta, cfg, seed)
  local self = setmetatable({}, Agent)
  self.cfg = {}
  for k, v in pairs(Agent.defaults) do self.cfg[k] = v end
  for k, v in pairs(cfg or {}) do self.cfg[k] = v end

  self.brain = BrainClass.new(net, 1.0)
  self.brain:seed(seed or 12345)
  self.sensory = Sensory.new(meta)
  self.motor = Motor.new(meta)

  self.heading = 0.0
  self.speed = self.cfg.baseSpeed
  self.yaw = 0.0
  self.steps = self.cfg.stepsPerFrame
  self.out = nil

  -- peak drive requested on each sensory channel this frame, in Hz: what the
  -- fly is being shown, as opposed to what it decides
  self.drive = { loom = 0, object = 0, bar = 0, flow = 0, landmark = 0 }
  self.loomLeft, self.loomRight = 0.0, 0.0

  -- populations sampled every frame for the overlay.  Sampled unconditionally
  -- so the rate estimate is continuous whether or not anything is drawing it.
  self.watch = {}
  self.watchRate = {}
  for _, name in ipairs({ "LC4", "LPLC2", "LC6", "LC11", "EPG", "Delta7", "PFL3" }) do
    local ids = meta.watch and meta.watch[name]
    if ids and #ids > 0 then
      self.watch[#self.watch + 1] = { name = name, ids = ids }
      self.watchRate[name] = 0.0
    end
  end
  return self
end

---------------------------------------------------------------------------
-- sense
---------------------------------------------------------------------------

-- Compress a looming stimulus into a firing rate.  LC4 responds to angular
-- size and to the rate at which it grows; the saturating form keeps a very
-- close object from driving the whole population into its refractory ceiling.
local function loomRate(cfg, angSize, closing, weight)
  local drive = weight * (0.35 * angSize + 0.11 * closing * angSize / 12.0)
  return cfg.loomPeak * (drive / (drive + 6.0))
end

function Agent:sense(p)
  local b, s, cfg = self.brain, self.sensory, self.cfg
  local drive = self.drive
  b:clearInput()
  drive.loom, drive.object, drive.bar = 0, 0, 0
  drive.flow, drive.landmark = 0, 0
  self.loomLeft, self.loomRight = 0.0, 0.0

  local function note(ch, hz)
    if hz > drive[ch] then drive[ch] = hz end
  end

  if p.landmarkAz then
    s:stimulate(b, "bar", p.landmarkAz, cfg.barSigma, cfg.barPeak)
    note("bar", cfg.barPeak)
    if cfg.landmarkPeak > 0.0 then
      s:stimulate(b, "landmark", p.landmarkAz, cfg.landmarkSigma, cfg.landmarkPeak)
      note("landmark", cfg.landmarkPeak)
    end
  end

  local n = 0
  local function loomAt(az, hz)
    s:stimulate(b, "loom", az, cfg.loomSigma, hz)
    note("loom", hz)
    -- which eye is being shouted at, for the overlay
    if az < 0 then
      if hz > self.loomLeft then self.loomLeft = hz end
    else
      if hz > self.loomRight then self.loomRight = hz end
    end
  end

  if p.threats then
    for i = 1, #p.threats do
      local t = p.threats[i]
      local hz = loomRate(cfg, t.angSize, t.closing or 0.0, t.weight or 1.0)
      if hz > 2.0 then
        loomAt(t.az, hz)
        s:stimulate(b, "object", t.az, cfg.objectSigma, hz * 0.45)
        note("object", hz * 0.45)
        n = n + 1
      end
    end
  end

  if p.objects then
    for i = 1, #p.objects do
      local o = p.objects[i]
      local hz = cfg.objectPeak * (o.angSize / (o.angSize + 14.0))
      if hz > 2.0 then
        s:stimulate(b, "object", o.az, cfg.objectSigma, hz)
        note("object", hz)
      end
    end
  end

  if p.walls then
    for i = 1, #p.walls do
      loomAt(p.walls[i], cfg.wallPeak)
      n = n + 1
    end
  end

  -- the fly's own motion regenerates optic flow, closing the loop
  s:flow(b, self.yaw, self.speed / cfg.maxSpeed)
  note("flow", 55.0 * math.abs(self.yaw) + 22.0 * (self.speed / cfg.maxSpeed))
  self.nLoom = n
end

---------------------------------------------------------------------------
-- think + act
---------------------------------------------------------------------------

function Agent:think(steps)
  self.brain:run(steps or self.steps)
end

function Agent:act()
  local cfg = self.cfg
  local out = self.motor:read(self.brain, self.steps)

  local turn = out.turn * cfg.turnGain
  if turn > cfg.maxTurn then turn = cfg.maxTurn
  elseif turn < -cfg.maxTurn then turn = -cfg.maxTurn end

  self.heading = self.heading + turn
  if self.heading > pi then self.heading = self.heading - 2 * pi
  elseif self.heading < -pi then self.heading = self.heading + 2 * pi end
  self.yaw = turn * 30.0   -- rad/s, fed back to the optic-flow channel

  self.escaping = out.escape > cfg.escapeThreshold
  local speed = cfg.baseSpeed + cfg.forwardGain * out.forward
  if self.escaping then
    speed = speed + cfg.escapeBoost * (out.escape - cfg.escapeThreshold) / 40.0
  end
  if speed > cfg.maxSpeed then speed = cfg.maxSpeed end
  self.speed = speed

  -- sample the watched populations every frame so their rate estimates stay
  -- continuous whether or not the overlay is being drawn
  for i = 1, #self.watch do
    local w = self.watch[i]
    self.watchRate[w.name] = self.brain:popRate(w.ids, self.steps)
  end

  self.out = out
  self.vx = cos(self.heading) * speed
  self.vy = sin(self.heading) * speed
  return out
end

-- One full cycle.  `budgetMs`/`elapsedMs` let the host throttle the number of
-- integration steps so the brain slows down instead of the game.
function Agent:update(percepts, elapsedMs, budgetMs)
  self:sense(percepts)
  self:think(self.steps)
  if elapsedMs and budgetMs then
    if elapsedMs > budgetMs and self.steps > self.cfg.minSteps then
      self.steps = self.steps - 2
    elseif elapsedMs < budgetMs * 0.55 and self.steps < self.cfg.stepsPerFrame then
      self.steps = self.steps + 1
    end
  end
  return self:act()
end

return Agent
