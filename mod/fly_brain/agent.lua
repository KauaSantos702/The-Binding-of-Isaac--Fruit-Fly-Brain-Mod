--[[ agent.lua -- the fly, with no Isaac API in sight.

  Holds the brain, the heading and the speed, and runs one sense -> think ->
  act cycle per call.  It takes a plain description of what is around the fly
  and returns a plain movement command, so exactly the same code path can be
  driven by the game (mod/main.lua) or by the offline arena
  (tools/behaviour.lua) that checks the fly actually behaves like a fly.

  percepts = {
    tracked  = { {az=, angSize=, weight=, prey=}, ... },  -- things worth going to
    threats  = { {az=, angSize=, closing=, weight=, sizeGain=, prey=, learned=}, ... },
    objects  = { {az=, angSize=}, ... },
    walls    = { {az=, dist=, closing=}, ... },     -- surfaces ahead
  }

  `closing` means different things on purpose.  For a threat it is how fast the
  *object itself* is coming -- the fly's own movement is not in it, because an
  animal that flies towards something is not being attacked by it.  For a wall
  it is how fast *she* is closing on the surface, because there the expansion
  really is hers to make, and that is the cue a fly avoids obstacles with.
  All azimuths are in degrees relative to the fly's current heading, positive
  to the fly's right.
]]

-- The Brain/Sensory/Motor modules are passed in by the host rather than
-- required here, because Isaac loads modules with include() and the offline
-- harness uses require().
local Agent = {}
Agent.__index = Agent

local cos, sin, exp, pi, atan = math.cos, math.sin, math.exp, math.pi, math.atan
local DEG = 180.0 / pi

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

  -- approach deceleration, keyed to how much of the visual field the target
  -- fills: full speed below landStartDeg, settled by landFullDeg
  landStartDeg    = 30.0,
  landFullDeg     = 80.0,
  landConeDeg     = 55.0,   -- must be roughly ahead to settle on
  landStop        = 0.96,   -- how much of the speed a full landing removes

  -- Riding prey.  She latches on once it fills this much of her view, holds on
  -- for feedFrames, then has had enough and spends disengageFrames getting
  -- clear before the next run.  At 30 fps that is a six second maul and a
  -- second and a half of backing off.
  feedSizeDeg     = 55.0,
  feedFrames      = 180,
  disengageFrames = 45,
  feedCalm        = 0.12,   -- how much the thing she is on still frightens her

  loomPeak        = 165.0,
  loomSigma       = 34.0,
  objectPeak      = 90.0,
  objectSigma     = 40.0,
  barPeak         = 55.0,   -- LC9/LC10: object position, reaches the premotor side
  barSigma        = 30.0,

  -- A surface she is flying into.  Unlike an object, the expansion here really
  -- is produced by her own movement, and that is the stimulus a fly avoids
  -- obstacles with -- so self-motion belongs in this one and only this one.
  -- Driven purely by expansion (sizeGain 0), so a wall she is not approaching
  -- is not frightening.
  --
  -- Three things keep it smooth.  The angular size comes from the measured
  -- distance, so there is no step.  `wallFade` takes the drive to zero at the
  -- edge of sensing range, so a probe coming into range does not arrive at
  -- full strength.  And the drive is weighted by how head-on the surface is,
  -- because obstacle avoidance is driven by expansion in front of the animal,
  -- not beside it.
  wallWeight      = 2.0,
  wallSpan        = 30.0,   -- px; half-width of the stretch treated as one object
  wallRange       = 110.0,  -- px; beyond this a surface is not sensed at all
  wallHeadOn      = 0.55,   -- how much a surface off to the side still counts
  -- A small standing-size term, so a surface close alongside her still
  -- registers when she is flying parallel to it and nothing is expanding.
  -- Far below the 0.35 an object gets: a wall is scenery, not a predator.
  wallSizeGain    = 0.04,

  -- The two ends of what she has learned.  An enemy sits on both the escape
  -- channel and the pursuit channel at once, and which one wins is the whole
  -- difference between prey and predator.  fly_brain/valence.lua moves these
  -- from experience; on their own they are just the timid baseline.
  huntWeight      = 0.45,   -- how much an enemy is worth to LC9/LC10
  preyThreat      = 1.00,   -- and how much of a threat that same enemy is

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
  -- defined before the first act(), so anything reading the fly's velocity on
  -- frame one gets a number rather than nil
  self.vx, self.vy = 0.0, 0.0
  self.feeding, self.feedFrames, self.satiatedFor = false, 0, 0

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
-- velocity and LPLC2 to expansion, so the motion term is what really drives
-- escape; `sizeGain` scales the standing-size term, which is our own
-- approximation and is set to zero for things that are merely large and still.
-- The saturating form keeps a very close object from pinning the whole
-- population at its refractory ceiling.
local function loomRate(cfg, angSize, closing, weight, sizeGain)
  sizeGain = sizeGain or 0.35
  local drive = weight * (sizeGain * angSize + 0.11 * closing * angSize / 12.0)
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

  -- Tracked targets go to LC9/LC10, the object-position cells.  This is the
  -- one channel measured to produce a *directed* turn toward a thing rather
  -- than away from it (tools/behaviour.lua, experiment 5) -- LC10a is the
  -- pathway a male fly tracks a female with.  Several targets can be present
  -- at once and they simply compete for the population.
  local primaryAz = p.landmarkAz
  self.trackSize, self.trackAz = 0.0, 0.0
  self.huntSize, self.huntAz = 0.0, 0.0
  if p.tracked then
    local best, bestPrey = -1.0, -1.0
    for i = 1, #p.tracked do
      local t = p.tracked[i]
      local sz = t.angSize or 14.0
      -- Prey is worth what her nerve says it is worth.  Frightened, an enemy
      -- barely registers here and she runs; bold, it is the thing she wants.
      local weight = t.weight or 1.0
      if t.prey then weight = weight * cfg.huntWeight end
      local hz = cfg.barPeak * weight * (sz / (sz + 12.0))
      if hz > 2.0 then
        s:stimulate(b, "bar", t.az, cfg.barSigma, hz)
        note("bar", hz)
        if not primaryAz then primaryAz = t.az end
      end
      -- remember the target filling the most of the visual field, for the
      -- approach deceleration in act()
      local salience = sz * weight
      if salience > best then
        best = salience
        self.trackSize, self.trackAz = sz, t.az
      end
      if t.prey and salience > bestPrey then
        bestPrey = salience
        self.huntSize, self.huntAz = sz, t.az
      end
    end
  end
  if p.landmarkAz then
    s:stimulate(b, "bar", p.landmarkAz, cfg.barSigma, cfg.barPeak)
    note("bar", cfg.barPeak)
  end
  if primaryAz and cfg.landmarkPeak > 0.0 then
    s:stimulate(b, "landmark", primaryAz, cfg.landmarkSigma, cfg.landmarkPeak)
    note("landmark", cfg.landmarkPeak)
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
      -- The same enemy is on both channels; her nerve turns one down as it
      -- turns the other up.  `learned` is what she remembers about this
      -- particular thing -- fire that burned her once reads worse afterwards.
      local weight = (t.weight or 1.0) * (t.learned or 1.0)
      if t.prey then
        weight = weight * cfg.preyThreat
        -- Riding it, the thing fills her whole view, and left alone that
        -- reads as the most enormous looming object she has ever seen -- she
        -- would scare herself straight back off her own kill.  Escape
        -- thresholds in flies are gated by what the animal is currently
        -- doing, and an animal settled on its food is not fleeing it.
        if self.feeding then weight = weight * cfg.feedCalm end
      end
      local hz = loomRate(cfg, t.angSize, t.closing or 0.0, weight, t.sizeGain)
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
      local w = p.walls[i]
      local d = w.dist
      if d and d > 1.0 and d < cfg.wallRange then
        local angSize = 2.0 * atan(cfg.wallSpan / d) * DEG
        -- fade to nothing at the edge of range, so nothing arrives in a step
        local fade = 1.0 - (d / cfg.wallRange) ^ 2
        -- and count a surface less the further off to the side it lies
        local head = cos(w.az * pi / 180.0)
        if head < 0.0 then head = 0.0 end
        head = cfg.wallHeadOn + (1.0 - cfg.wallHeadOn) * head
        local hz = loomRate(cfg, angSize, w.closing,
                            cfg.wallWeight * fade * head, cfg.wallSizeGain)
        if hz > 2.0 then
          loomAt(w.az, hz)
          n = n + 1
        end
      end
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

  -- Approach deceleration.  A landing fly slows in proportion to how fast the
  -- target is expanding across its retina, which is why it arrives at a
  -- surface at nearly zero speed instead of hitting it (Borst 1986; van
  -- Breugel & Dickinson 2012).  We use the target's angular size as the proxy.
  --
  -- This does not come out of the connectome -- it belongs with the ventral
  -- nerve cord assumptions listed in motor.lua.  Without it the fly has a
  -- floor speed and can only circle a thing it is trying to settle on.
  local fear = out.escape / cfg.escapeThreshold
  if fear > 1.0 then fear = 1.0 end

  -- Feeding.  A fly does not strafe its food: it lands on it and stays, then
  -- has enough and lifts off.  Once prey fills `feedSizeDeg` of her view she
  -- is on it, and she rides it until she is satiated, which is when the
  -- approach starts again from the top.
  --
  -- The cone below does not apply here.  It stops her settling on something
  -- beside her while she is still flying at it, but once she is actually on a
  -- thing the bearing swings through every angle and the test would throw her
  -- straight back off.
  --
  -- Satiation is not in the connectome -- the model has no notion of having
  -- had enough -- so it sits here with the landing rule, for the same reason.
  local onPrey = self.huntSize >= cfg.feedSizeDeg and fear < 0.9
  if self.satiatedFor > 0 then
    self.satiatedFor = self.satiatedFor - 1
    self.feeding = false
  elseif onPrey then
    self.feedFrames = self.feedFrames + 1
    self.feeding = true
    if self.feedFrames >= cfg.feedFrames then
      self.satiatedFor = cfg.disengageFrames
      self.feedFrames = 0
      self.feeding = false
    end
  else
    self.feedFrames = self.feedFrames > 0 and self.feedFrames - 1 or 0
    self.feeding = false
  end

  local land = 0.0
  if self.feeding then
    land = 1.0
  elseif self.satiatedFor > 0 then
    land = 0.0                       -- she has had enough; let her get away
  elseif self.trackSize > cfg.landStartDeg then
    land = (self.trackSize - cfg.landStartDeg)
           / (cfg.landFullDeg - cfg.landStartDeg)
    if land > 1.0 then land = 1.0 end
    -- only if it is roughly in front: a fly does not land on what is beside it
    local off = self.trackAz
    if off < 0 then off = -off end
    if off > cfg.landConeDeg then land = 0.0 end
    -- and never while frightened
    land = land * (1.0 - fear)
  end
  self.landing = land
  speed = speed * (1.0 - land * cfg.landStop)

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
