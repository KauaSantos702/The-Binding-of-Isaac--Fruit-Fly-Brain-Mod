--[[ Connectome -- a familiar fly flown by a real fly's brain.

  Every frame this mod builds a visual scene from the room, injects it as
  Poisson spike trains into 2053 real Drosophila neurons wired by 83791 real
  synapses from the FlyWire whole-brain connectome, runs 33 ms of leaky
  integrate-and-fire dynamics, and reads the fly's next move out of its
  descending neurons.

  There is no state machine and no behaviour tree.  Where the fly goes is
  whatever DNa02, DNp09, DNp01 and the rest happen to be doing.

  This file does only two jobs: turn the room into percepts, and turn the
  agent's movement command into a Velocity.  Everything else lives in
  fly_brain/, which has no Isaac API in it and is exercised offline by
  tools/behaviour.lua and tools/bench.lua.

  See docs/METHODS.md for how the circuit was extracted and validated.
]]

local mod = RegisterMod("Connectome", 1)

local Loader  = include("fly_brain.loader")
local Brain   = include("fly_brain.brain")
local Sensory = include("fly_brain.sensory")
local Motor   = include("fly_brain.motor")
local Agent   = include("fly_brain.agent")
local Hud     = include("fly_brain.hud")

local ITEM = Isaac.GetItemIdByName("Connectome")
local FLY_VARIANT = Isaac.GetEntityVariantByName("Connectome Fly")

-- TEST MODE ---------------------------------------------------------------
-- While this is true the item is handed to you at the start of every run and
-- the neuron overlay starts switched on, so the mod can be tried without the
-- debug console.  Unlike the console, AddCollectible does not disable
-- achievements.  Set it to false to play normally and find the item in a
-- Treasure Room.
local TEST_MODE = true
----------------------------------------------------------------------------

-- Only the game-facing feel lives here; nothing in this block changes what
-- the neurons compute, only how the room is presented and how the result is
-- scaled into pixels.
local CFG = {
  budgetMs       = 6.0,    -- simulation time allowed per fly per frame
  visionRange    = 260.0,  -- px; beyond this an entity is not resolved
  wallProbe      = 46.0,   -- px ahead to test for walls
  leashRadius    = 190.0,  -- px from the player before we pull the fly back
  leashGain      = 0.10,   -- a game constraint, not biology -- see README
  damageCooldown = 12,     -- frames between hits on the same enemy
  contactReach   = 13.0,
}

---------------------------------------------------------------------------
-- one shared network, one brain per fly
---------------------------------------------------------------------------
local net, meta, loadError

-- Isaac's include() resolves paths relative to the mod that is *currently
-- being loaded*, so it only works while main.lua is executing.  Called later
-- from inside a callback it returns nil without raising, which is why the
-- connectome has to be pulled in here at load time rather than on first use.
local function loadNetwork()
  if net or loadError then return net end
  local ok, err = pcall(function()
    -- The data files both return their table and assign it to a global, so a
    -- script loader that drops the return value still works.
    local packed = include("fly_brain.net_csr")
    if type(packed) ~= "table" or not packed.parts then packed = FlyBrainNetCSR end
    meta = include("fly_brain.net_meta")
    if type(meta) ~= "table" or not meta.sensory then meta = FlyBrainNetMeta end
    local vrest = include("fly_brain.net_vrest")
    if type(vrest) ~= "table" or #vrest == 0 then vrest = FlyBrainNetVrest end

    if type(packed) ~= "table" then error("net_csr gave " .. type(packed)) end
    if not packed.parts then error("net_csr has no part manifest") end
    if not meta or not meta.sensory then error("net_meta has no sensory map") end
    if type(vrest) ~= "table" then error("net_vrest gave " .. type(vrest)) end

    net = Loader.load(packed, include)
    net.vrest = vrest
  end)
  if not ok then
    loadError = tostring(err)
    Isaac.DebugString("[Connectome] failed to load the connectome: " .. loadError)
    return nil
  end
  Isaac.DebugString(string.format("[Connectome] loaded %d neurons / %d synapses",
    net.n, net.nnz))
  return net
end

---------------------------------------------------------------------------
-- geometry
---------------------------------------------------------------------------
local atan, cos, sin, sqrt, pi = math.atan, math.cos, math.sin, math.sqrt, math.pi
local DEG = 180.0 / pi

-- Signed angle in degrees from the fly's heading to a world offset, positive
-- to the fly's right.  Isaac's Y axis points down, so a positive cross
-- product is a clockwise (rightward) offset on screen.
local function relAzimuth(hx, hy, dx, dy)
  return atan(hx * dy - hy * dx, hx * dx + hy * dy) * DEG
end

---------------------------------------------------------------------------
-- the room, as the fly sees it
---------------------------------------------------------------------------
local function buildPercepts(fly, agent, player)
  local room = Game():GetRoom()
  local pos = fly.Position
  local hx, hy = cos(agent.heading), sin(agent.heading)
  local range2 = CFG.visionRange * CFG.visionRange

  local threats, objects, walls = {}, {}, {}

  local dp = player.Position - pos
  local landmarkAz = relAzimuth(hx, hy, dp.X, dp.Y)

  for _, ent in ipairs(Isaac.GetRoomEntities()) do
    local d = ent.Position - pos
    local d2 = d.X * d.X + d.Y * d.Y
    if d2 < range2 and d2 > 1.0 then
      local weight, isObject = 0.0, false
      if ent:IsVulnerableEnemy() then
        weight = 1.0
      elseif ent.Type == EntityType.ENTITY_PROJECTILE then
        weight = 1.4
      elseif ent.Type == EntityType.ENTITY_BOMB then
        weight = 1.6
      elseif ent.Type == EntityType.ENTITY_PICKUP then
        isObject = true
      end

      if weight > 0.0 or isObject then
        local dist = sqrt(d2)
        local radius = ent.Size > 0 and ent.Size or 8.0
        local az = relAzimuth(hx, hy, d.X, d.Y)
        local angSize = 2.0 * atan(radius / dist) * DEG
        if isObject then
          objects[#objects + 1] = { az = az, angSize = angSize }
        else
          -- how fast the object is closing, along the line of sight
          local v = ent.Velocity
          local closing = 0.0
          if v then
            closing = -((v.X - fly.Velocity.X) * d.X
                      + (v.Y - fly.Velocity.Y) * d.Y) / dist
          end
          if closing < 0.0 then closing = 0.0 end
          threats[#threats + 1] =
            { az = az, angSize = angSize, closing = closing, weight = weight }
        end
      end
    end
  end

  -- Walls loom too; a fly that cannot see them grinds along them.
  for _, probe in ipairs({ 0, 40, -40 }) do
    local a = agent.heading + probe / DEG
    local ahead = pos + Vector(cos(a) * CFG.wallProbe, sin(a) * CFG.wallProbe)
    if not room:IsPositionInRoom(ahead, 0) then
      walls[#walls + 1] = probe
    end
  end

  return { landmarkAz = landmarkAz, threats = threats,
           objects = objects, walls = walls }
end

---------------------------------------------------------------------------
-- movement + contact damage
---------------------------------------------------------------------------
local function applyMovement(fly, agent, player)
  local vx, vy = agent.vx, agent.vy

  -- Leash.  Not biology: an Isaac familiar has to stay useful to the player,
  -- and one room is not a fly's whole world.  Engages only past leashRadius.
  local d = player.Position - fly.Position
  local dist = d:Length()
  if dist > CFG.leashRadius then
    local pull = CFG.leashGain * (dist - CFG.leashRadius) / 60.0
    if pull > 1.0 then pull = 1.0 end
    vx = vx + d.X / dist * agent.speed * pull * 2.0
    vy = vy + d.Y / dist * agent.speed * pull * 2.0
    agent.heading = atan(vy, vx)
  end

  fly.Velocity = Vector(vx, vy)

  local sprite = fly:GetSprite()
  sprite.Rotation = agent.heading * DEG + 90.0  -- the sprite is drawn facing up
  if not sprite:IsPlaying("Fly") then sprite:Play("Fly", false) end
end

local function damageContact(fly, data, agent, player)
  data.hitCooldown = data.hitCooldown or {}
  local dmg = 2.0 + player.Damage * 0.55
  if agent.escaping then dmg = dmg * 1.6 end
  local reach = CFG.contactReach + fly.Size

  for _, ent in ipairs(Isaac.GetRoomEntities()) do
    if ent:IsVulnerableEnemy() then
      local key = GetPtrHash(ent)
      local cd = data.hitCooldown[key] or 0
      if cd > 0 then
        data.hitCooldown[key] = cd - 1
      elseif ent.Position:Distance(fly.Position) < reach + ent.Size then
        ent:TakeDamage(dmg, 0, EntityRef(fly), 0)
        data.hitCooldown[key] = CFG.damageCooldown
      end
    end
  end
end

---------------------------------------------------------------------------
-- callbacks
---------------------------------------------------------------------------
function mod:onCache(player, flag)
  if flag == CacheFlag.CACHE_FAMILIARS then
    local count = player:GetCollectibleNum(ITEM)
    player:CheckFamiliar(FLY_VARIANT, count, player:GetCollectibleRNG(ITEM),
      Isaac.GetItemConfig():GetCollectible(ITEM))
  end
end

function mod:onFamiliarInit(fly)
  fly.IsFollower = false
  fly:GetSprite():Play("Fly", true)
end

local lastCostMs = 0.0

function mod:onFamiliarUpdate(fly)
  if not net then return end

  local data = fly:GetData()
  local player = fly.Player or Isaac.GetPlayer(0)

  if not data.agent then
    data.agent = Agent.new(Brain, Sensory, Motor, net, meta, nil,
      fly.InitSeed ~ 0x5bf03635)
    data.agent.heading = (fly.InitSeed % 6283) / 1000.0 - pi
    fly.Position = player.Position + Vector(24, 0)
  end
  local agent = data.agent

  local percepts = buildPercepts(fly, agent, player)

  local t0 = Isaac.GetTime()
  agent:sense(percepts)
  agent:think(agent.steps)
  local elapsed = Isaac.GetTime() - t0
  lastCostMs = lastCostMs * 0.9 + elapsed * 0.1

  -- Adaptive budget: when several flies share a frame each one runs fewer
  -- integration steps, so the brain slows down instead of the game.
  if elapsed > CFG.budgetMs and agent.steps > agent.cfg.minSteps then
    agent.steps = agent.steps - 2
  elseif elapsed < CFG.budgetMs * 0.55 and agent.steps < agent.cfg.stepsPerFrame then
    agent.steps = agent.steps + 1
  end

  agent:act()
  applyMovement(fly, agent, player)
  damageContact(fly, data, agent, player)
end

---------------------------------------------------------------------------
-- debug overlay: type `flybrain` in the console to toggle
---------------------------------------------------------------------------
local showDebug = TEST_MODE
local compassOn = false

local function eachAgent(fn)
  for _, fly in ipairs(Isaac.FindByType(EntityType.ENTITY_FAMILIAR, FLY_VARIANT, -1)) do
    local agent = fly:GetData().agent
    if agent then fn(agent) end
  end
end

local function say(msg)
  Isaac.ConsoleOutput(msg .. "\n")
  return msg
end

function mod:onCommand(cmd, params)
  if cmd ~= "flybrain" then return end
  params = (params or ""):lower()

  if params == "compass" then
    -- Switch on the MeTu -> TuBu -> ER -> EPG landmark pathway.  It is off by
    -- default because it costs about half the simulation's CPU and the ring
    -- attractor produces no behaviour in this model; see docs/METHODS.md.
    compassOn = not compassOn
    eachAgent(function(a) a.cfg.landmarkPeak = compassOn and 110.0 or 0.0 end)
    return say("landmark pathway (MeTu->TuBu->ER->EPG) "
      .. (compassOn and "on -- costs roughly 2x CPU" or "off"))
  elseif params == "reset" then
    eachAgent(function(a) a.brain:reset() end)
    return say("brains reset to rest")
  end

  showDebug = not showDebug
  return say("connectome overlay " .. (showDebug and "on" or "off")
    .. "  (also: flybrain compass, flybrain reset)")
end

-- history buffers for the scrolling graphs, one set per fly
local histories = {}

local function historyFor(key)
  local h = histories[key]
  if not h then
    h = { active = Hud.History.new(110), escape = Hud.History.new(110),
          danger = Hud.History.new(110) }
    histories[key] = h
  end
  return h
end

-- 0..1 "how strongly is the fly feeling this", for the mood bars
local function unit(v, full)
  v = v / full
  if v < 0 then return 0 elseif v > 1 then return 1 end
  return v
end

function mod:onRender()
  if not showDebug then return end

  local x, y = 24, 36
  if loadError then
    Hud.text("connectome FAILED to load:", x, y, 1.0, 0.4, 0.4)
    -- wrap, so the whole message is readable on screen
    local msg = loadError:gsub("^.*fly%-brain%-connectome/", "")
    for k = 0, 3 do
      local piece = msg:sub(k * 58 + 1, (k + 1) * 58)
      if piece == "" then break end
      Hud.text(piece, x, y + 12 + k * 11, 1.0, 0.6, 0.6)
    end
    return
  end
  if not net then return end

  Hud.rect(x - 8, y - 8, 300, 214, 0.04, 0.05, 0.07, 0.72)
  Hud.text(string.format("CONNECTOME  %d neurons / %d synapses", net.n, net.nnz),
    x, y, 0.6, 0.95, 1.0)
  y = y + 14

  local flies = Isaac.FindByType(EntityType.ENTITY_FAMILIAR, FLY_VARIANT, -1)
  if #flies == 0 then
    Hud.text("no fly in the room", x, y, 0.8, 0.8, 0.5)
    return
  end

  local fly = flies[1]
  local agent = fly:GetData().agent
  if not agent or not agent.out then
    Hud.text("waiting for the first brain update...", x, y, 0.8, 0.8, 0.5)
    return
  end

  local o, d = agent.out, agent.drive
  local h = historyFor(GetPtrHash(fly))
  local active = agent.brain:activeCount()
  h.active:push(active)
  h.escape:push(o.escape)
  h.danger:push(d.loom)

  -- ---- what the fly is sensing -----------------------------------------
  Hud.text("SENSING", x, y, 0.55, 0.8, 0.9)
  y = y + 13
  local bx, bw, bh = x + 78, 150, 9
  Hud.bar(bx, y, bw, bh, unit(d.loom, 165), "looming", { 1.0, 0.45, 0.35 })
  y = y + 13
  Hud.biBar(bx, y, bw, bh, (agent.loomRight - agent.loomLeft) / 165.0,
    "which eye", { 0.45, 0.8, 1.0 }, { 1.0, 0.7, 0.35 },
    agent.loomLeft > agent.loomRight and "LEFT"
      or (agent.loomRight > agent.loomLeft and "RIGHT" or "--"))
  y = y + 13
  Hud.bar(bx, y, bw, bh, unit(d.object, 90), "objects", { 0.9, 0.85, 0.4 })
  y = y + 13
  Hud.bar(bx, y, bw, bh, unit(d.flow, 77), "self-motion", { 0.5, 0.85, 0.6 })
  y = y + 17

  -- ---- what the brain is doing about it ---------------------------------
  Hud.text("MOOD", x, y, 0.55, 0.8, 0.9)
  y = y + 13
  local fear = unit(o.escape, 45)
  Hud.bar(bx, y, bw, bh, fear, "fear", { 1.0, 0.3, 0.3 },
    agent.escaping and string.format("%d  FLEE", fear * 100 + 0.5) or nil)
  y = y + 13
  Hud.bar(bx, y, bw, bh, unit(active, 900), "arousal", { 0.8, 0.5, 1.0 })
  y = y + 13
  Hud.bar(bx, y, bw, bh, unit(o.forward, 60), "drive to move", { 0.5, 0.9, 0.9 })
  y = y + 13
  Hud.biBar(bx, y, bw, bh, o.turn / 12.0, "turning",
    { 0.45, 0.8, 1.0 }, { 1.0, 0.7, 0.35 },
    string.format("%+.1f", o.turn))
  y = y + 13
  Hud.biBar(bx, y, bw, bh, (o.dnR - o.dnL) / 40.0, "brain L/R",
    { 0.45, 0.8, 1.0 }, { 1.0, 0.7, 0.35 },
    string.format("%.0f/%.0f", o.dnL, o.dnR))
  y = y + 20

  -- ---- populations, and the last few seconds ----------------------------
  local entries = {}
  for i = 1, #agent.watch do
    local name = agent.watch[i].name
    entries[#entries + 1] = { name = name, hz = agent.watchRate[name] or 0 }
  end
  entries[#entries + 1] = { name = "DN-L", hz = o.dnL }
  entries[#entries + 1] = { name = "DN-R", hz = o.dnR }
  Hud.popRow(x, y, 24, 26, entries, 60)
  y = y + 40

  Hud.graph(x, y, 132, 30, h.active, 900, { 0.8, 0.5, 1.0 }, "neurons firing")
  Hud.graph(x + 150, y, 132, 30, h.escape, 45, { 1.0, 0.3, 0.3 }, "fear")
  y = y + 34

  Hud.text(string.format("%d active   %d steps   %.1f ms/frame   heading %+.0f",
    active, agent.steps, lastCostMs, agent.heading * DEG), x, y, 0.6, 0.7, 0.75)
end

function mod:onGameStarted()
  if not TEST_MODE then return end
  local player = Isaac.GetPlayer(0)
  if player and not player:HasCollectible(ITEM) then
    player:AddCollectible(ITEM, 0, false)
    Isaac.DebugString("[Connectome] TEST_MODE: granted item " .. tostring(ITEM))
  end
end

-- Pull the connectome in now, while include() still resolves (see loadNetwork).
loadNetwork()

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, mod.onGameStarted)
mod:AddCallback(ModCallbacks.MC_EVALUATE_CACHE, mod.onCache)
mod:AddCallback(ModCallbacks.MC_FAMILIAR_INIT, mod.onFamiliarInit, FLY_VARIANT)
mod:AddCallback(ModCallbacks.MC_FAMILIAR_UPDATE, mod.onFamiliarUpdate, FLY_VARIANT)
mod:AddCallback(ModCallbacks.MC_EXECUTE_CMD, mod.onCommand)
mod:AddCallback(ModCallbacks.MC_POST_RENDER, mod.onRender)

Isaac.DebugString("[Connectome] registered (item " .. tostring(ITEM) ..
  ", familiar variant " .. tostring(FLY_VARIANT) ..
  ", network " .. (net and "ready" or ("FAILED: " .. tostring(loadError))) .. ")")
