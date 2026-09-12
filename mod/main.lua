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
  leashRamp      = 130.0,  -- px over which the pull reaches full strength
  leashSpeed     = 3.2,    -- a game constraint, not biology -- see README
  damageCooldown = 12,     -- frames between hits on the same enemy
  contactReach   = 13.0,

  -- Where the boundary between the small-object cells and the looming cells
  -- sits.  Below smallDeg of angular size, and expanding slower than
  -- expansionDeg per frame, a thing is not an escape stimulus at all.
  smallDeg       = 6.0,
  loomDeg        = 18.0,
  expansionDeg   = 0.40,
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
-- grid entities: poop, rocks, fires.  These do not appear in
-- Isaac.GetRoomEntities(), so without this scan the fly cannot see them at
-- all -- and poop is the one thing in this game a real fly is built for.
---------------------------------------------------------------------------
local GRID_KIND = {}
do
  local function kind(name, k)
    local t = GridEntityType[name]
    if t then GRID_KIND[t] = k end
  end
  kind("GRID_POOP", "track")          -- fermenting organic matter: go to it
  kind("GRID_FIREPLACE", "fire")
  for _, r in ipairs({ "GRID_ROCK", "GRID_ROCKB", "GRID_ROCKT", "GRID_ROCK_ALT",
                       "GRID_ROCK_SS", "GRID_ROCK_SPIKED", "GRID_ROCK_GOLD",
                       "GRID_ROCK_BOMB", "GRID_PILLAR", "GRID_STATUE",
                       "GRID_TNT", "GRID_SPIKES", "GRID_SPIKES_ONOFF" }) do
    kind(r, "object")                 -- large and motionless: seen, not feared
  end
end

local gridCache = { frame = -1000, room = -1, items = {} }
local GRID_RESCAN = 15                -- frames; poop gets destroyed, fires go out

local function scanGrid()
  local game = Game()
  local frame = game:GetFrameCount()
  local roomIdx = game:GetLevel():GetCurrentRoomIndex()
  if gridCache.room == roomIdx and frame - gridCache.frame < GRID_RESCAN then
    return gridCache.items
  end

  local room = game:GetRoom()
  local items = {}
  for i = 0, room:GetGridSize() - 1 do
    local g = room:GetGridEntity(i)
    if g then
      local k = GRID_KIND[g.Desc.Type]
      -- a fully destroyed grid entity stays in the list with State 1000
      if k and g.State < 1000 then
        items[#items + 1] = { pos = g.Position, kind = k }
      end
    end
  end

  gridCache.room = roomIdx
  gridCache.frame = frame
  gridCache.items = items
  return items
end

---------------------------------------------------------------------------
-- the room, as the fly sees it
---------------------------------------------------------------------------
local function buildPercepts(fly, agent, player)
  local room = Game():GetRoom()
  local pos = fly.Position
  local hx, hy = cos(agent.heading), sin(agent.heading)
  local range2 = CFG.visionRange * CFG.visionRange

  local threats, objects, walls, tracked = {}, {}, {}, {}

  -- ---- the player ------------------------------------------------------
  -- A fly lands on a motionless animal and bolts the moment it moves, because
  -- LC4 is tuned to angular velocity and LPLC2 to expansion: something merely
  -- large and still produces neither.  So Isaac goes on the tracking channel
  -- always, and on the looming channel only in proportion to how fast he is
  -- moving.  Stand still and she will come to you.
  local dp = player.Position - pos
  local pdist = dp:Length()
  if pdist > 1.0 then
    local pAz = relAzimuth(hx, hy, dp.X, dp.Y)
    local pSize = 2.0 * atan(14.0 / math.max(pdist, 10.0)) * DEG
    tracked[#tracked + 1] = { az = pAz, angSize = pSize, weight = 1.0 }

    local pv = player.Velocity
    local closing = -((pv.X - fly.Velocity.X) * dp.X
                    + (pv.Y - fly.Velocity.Y) * dp.Y) / pdist
    if closing < 0.0 then closing = 0.0 end
    local startle = closing + 0.35 * pv:Length()
    if startle > 0.4 then
      threats[#threats + 1] = { az = pAz, angSize = pSize, closing = startle,
                                weight = 0.9, sizeGain = 0.0 }
    end
  end

  -- ---- grid entities ---------------------------------------------------
  for _, item in ipairs(scanGrid()) do
    local g = item.pos - pos
    local gd2 = g.X * g.X + g.Y * g.Y
    if gd2 < range2 and gd2 > 1.0 then
      local gdist = sqrt(gd2)
      local az = relAzimuth(hx, hy, g.X, g.Y)
      local angSize = 2.0 * atan(13.0 / gdist) * DEG
      if item.kind == "track" then
        -- poop outranks the player: a fly would rather be on the poop
        tracked[#tracked + 1] = { az = az, angSize = angSize, weight = 1.35 }
      elseif item.kind == "fire" then
        threats[#threats + 1] = { az = az, angSize = angSize,
                                  closing = 3.0, weight = 1.5 }
      else
        objects[#objects + 1] = { az = az, angSize = angSize }
      end
    end
  end

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
      elseif ent.Type == EntityType.ENTITY_FIREPLACE then
        weight = 1.5                  -- heat and flicker: a fly keeps away
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

          -- Which pathway does this belong on?  A fly's escape circuit does
          -- not recognise anything: LC4 answers to angular velocity and LPLC2
          -- to expansion, while small slow things are handled by a different
          -- set of cells (LC11/LC17/LC18) that drive no escape at all.  So the
          -- split is made on the stimulus, not on what kind of entity it is --
          -- which is why a distant Attack Fly is ignored and a tear, small but
          -- expanding fast, is not.
          local bySize = (angSize - CFG.smallDeg) / (CFG.loomDeg - CFG.smallDeg)
          local byExpansion = (angSize * closing / dist) / CFG.expansionDeg
          local loomness = bySize > byExpansion and bySize or byExpansion
          if loomness < 0.0 then loomness = 0.0 elseif loomness > 1.0 then loomness = 1.0 end

          if loomness > 0.02 then
            threats[#threats + 1] = { az = az, angSize = angSize,
                                      closing = closing,
                                      weight = weight * loomness }
          end
          if loomness < 0.98 then
            objects[#objects + 1] = { az = az,
                                      angSize = angSize * (1.0 - loomness) }
          end
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

  return { tracked = tracked, threats = threats,
           objects = objects, walls = walls }
end

---------------------------------------------------------------------------
-- movement + contact damage
---------------------------------------------------------------------------
local function applyMovement(fly, agent, player)
  local vx, vy = agent.vx, agent.vy

  -- Leash.  Not biology: an Isaac familiar has to stay useful to the player,
  -- and one room is not a fly's whole world.  Its strength depends only on
  -- distance, so that a fly which has settled on something close by is left
  -- alone, while one that has wandered off is still brought back.
  local d = player.Position - fly.Position
  local dist = d:Length()
  if dist > CFG.leashRadius then
    local pull = (dist - CFG.leashRadius) / CFG.leashRamp
    if pull > 1.0 then pull = 1.0 end
    vx = vx + d.X / dist * CFG.leashSpeed * pull
    vy = vy + d.Y / dist * CFG.leashSpeed * pull
    if pull > 0.25 then agent.heading = atan(vy, vx) end
  end

  fly.Velocity = Vector(vx, vy)

  local sprite = fly:GetSprite()
  sprite.Rotation = agent.heading * DEG + 90.0  -- the sprite is drawn facing up
  -- settled on something: slow the wingbeat rather than hovering at full tilt
  local anim = (agent.landing or 0) > 0.6 and "Idle" or "Fly"
  if not sprite:IsPlaying(anim) then sprite:Play(anim, false) end
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
-- HUD modes: off, a small read-out at the bottom right, or the full panel.
local HUD_OFF, HUD_COMPACT, HUD_FULL = 0, 1, 2
local hudMode = TEST_MODE and HUD_COMPACT or HUD_OFF
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

  hudMode = (hudMode + 1) % 3
  local names = { [0] = "off", [1] = "compact", [2] = "full" }
  return say("connectome overlay: " .. names[hudMode]
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

-- palette, kept close to the glowing connectome on the item sprite
local C_DIM   = { 0.55, 0.60, 0.66 }
local C_LABEL = { 0.45, 0.78, 0.88 }
local C_FEAR  = { 0.95, 0.30, 0.32 }
local C_LEFT  = { 0.40, 0.72, 1.00 }
local C_RIGHT = { 1.00, 0.68, 0.32 }
local C_LOOM  = { 1.00, 0.50, 0.38 }
local C_BUSY  = { 0.70, 0.50, 1.00 }

local function firstAgent()
  local flies = Isaac.FindByType(EntityType.ENTITY_FAMILIAR, FLY_VARIANT, -1)
  for i = 1, #flies do
    local a = flies[i]:GetData().agent
    if a and a.out then return a, flies[i] end
  end
end

---------------------------------------------------------------------------
-- compact read-out: bottom right, roughly the footprint of a HUD counter
---------------------------------------------------------------------------
local function renderCompact(agent)
  local o = agent.out
  local w = 104
  local x = Isaac.GetScreenWidth() - w - 14
  local y = Isaac.GetScreenHeight() - 40

  Hud.stext(string.format("%d neurons", net.n), x, y, 0.5, C_LABEL)
  Hud.stext(string.format("%d synapses", net.nnz), x + 52, y, 0.5, C_DIM)

  -- which half of the brain is shouting: this is the decision to turn,
  -- forming in front of you
  Hud.thinBiBar(x, y + 10, w, 4, (o.dnR - o.dnL) / 40.0, C_LEFT, C_RIGHT)

  -- fear, which flashes while the escape neurons are over threshold
  local fear = unit(o.escape, 45)
  local a = 1.0
  if agent.escaping then
    a = 0.65 + 0.35 * math.sin(Isaac.GetFrameCount() * 0.45)
  end
  Hud.thinBar(x, y + 17, w, 4, fear, C_FEAR, a)

  -- how much of the brain is awake right now
  Hud.thinBar(x, y + 24, w, 3, unit(agent.brain:activeCount(), 900), C_BUSY, 0.75)

  if agent.escaping then
    Hud.stext("FLEE", x + w - 18, y + 29, 0.45, C_FEAR, a)
  end
end

---------------------------------------------------------------------------
-- full read-out: everything, for tuning and for watching it think
---------------------------------------------------------------------------
local function renderFull(agent, fly)
  local x, y = 24, 36
  local o, d = agent.out, agent.drive
  local h = historyFor(GetPtrHash(fly))
  local active = agent.brain:activeCount()
  h.active:push(active)
  h.escape:push(o.escape)
  h.danger:push(d.loom)

  Hud.rect(x - 8, y - 8, 300, 214, 0.04, 0.05, 0.07, 0.72)
  Hud.text(string.format("%d neurons / %d synapses", net.n, net.nnz),
    x, y, C_LABEL[1], C_LABEL[2], C_LABEL[3])
  y = y + 14

  local bx, bw, bh = x + 78, 150, 9
  Hud.text("SENSING", x, y, 0.55, 0.8, 0.9)
  y = y + 13
  Hud.bar(bx, y, bw, bh, unit(d.loom, 165), "looming", C_LOOM)
  y = y + 13
  Hud.biBar(bx, y, bw, bh, (agent.loomRight - agent.loomLeft) / 165.0,
    "which eye", C_LEFT, C_RIGHT,
    agent.loomLeft > agent.loomRight and "LEFT"
      or (agent.loomRight > agent.loomLeft and "RIGHT" or "--"))
  y = y + 13
  Hud.bar(bx, y, bw, bh, unit(d.object, 90), "objects", { 0.9, 0.85, 0.4 })
  y = y + 13
  Hud.bar(bx, y, bw, bh, unit(d.flow, 77), "self-motion", { 0.5, 0.85, 0.6 })
  y = y + 17

  Hud.text("MOOD", x, y, 0.55, 0.8, 0.9)
  y = y + 13
  local fear = unit(o.escape, 45)
  Hud.bar(bx, y, bw, bh, fear, "fear", C_FEAR,
    agent.escaping and string.format("%d  FLEE", fear * 100 + 0.5) or nil)
  y = y + 13
  Hud.bar(bx, y, bw, bh, unit(active, 900), "arousal", C_BUSY)
  y = y + 13
  Hud.bar(bx, y, bw, bh, unit(o.forward, 60), "drive to move", { 0.5, 0.9, 0.9 })
  y = y + 13
  Hud.biBar(bx, y, bw, bh, o.turn / 12.0, "turning", C_LEFT, C_RIGHT,
    string.format("%+.1f", o.turn))
  y = y + 13
  Hud.biBar(bx, y, bw, bh, (o.dnR - o.dnL) / 40.0, "brain L/R", C_LEFT, C_RIGHT,
    string.format("%.0f/%.0f", o.dnL, o.dnR))
  y = y + 20

  local entries = {}
  for i = 1, #agent.watch do
    local name = agent.watch[i].name
    entries[#entries + 1] = { name = name, hz = agent.watchRate[name] or 0 }
  end
  entries[#entries + 1] = { name = "DN-L", hz = o.dnL }
  entries[#entries + 1] = { name = "DN-R", hz = o.dnR }
  Hud.popRow(x, y, 24, 26, entries, 60)
  y = y + 40

  Hud.graph(x, y, 132, 30, h.active, 900, C_BUSY, "neurons firing")
  Hud.graph(x + 150, y, 132, 30, h.escape, 45, C_FEAR, "fear")
  y = y + 34

  Hud.text(string.format("%d active   %d steps   %.1f ms/frame   heading %+.0f",
    active, agent.steps, lastCostMs, agent.heading * DEG), x, y, 0.6, 0.7, 0.75)
end

function mod:onRender()
  if hudMode == HUD_OFF then return end

  if loadError then
    Hud.text("connectome FAILED to load:", 24, 36, 1.0, 0.4, 0.4)
    local msg = loadError:gsub("^.*fly%-brain%-connectome/", "")
    for k = 0, 3 do
      local piece = msg:sub(k * 58 + 1, (k + 1) * 58)
      if piece == "" then break end
      Hud.text(piece, 24, 48 + k * 11, 1.0, 0.6, 0.6)
    end
    return
  end
  if not net then return end

  local agent, fly = firstAgent()
  if not agent then return end

  if hudMode == HUD_FULL then
    renderFull(agent, fly)
  else
    renderCompact(agent)
  end
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
