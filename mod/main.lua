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
local Valence = include("fly_brain.valence")

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
  -- Wall sensing: which directions to probe relative to her heading, and how
  -- finely.  How that distance becomes a stimulus lives in agent.lua with the
  -- rest of the sensory rules.
  wallProbes     = { 0, 35, -35, 70, -70 },
  wallRange      = 110.0,
  wallSteps      = 8,
  leashRadius    = 190.0,  -- px from the player before we pull the fly back
  leashRamp      = 130.0,  -- px over which the pull reaches full strength
  leashSpeed     = 3.2,    -- a game constraint, not biology -- see README
  damageCooldown = 12,     -- frames between hits on the same enemy
  contactReach   = 13.0,
  feedReach      = 26.0,   -- px; close enough to be riding an enemy
  feedCling      = 0.30,   -- how hard she holds on to its centre
  burnRadius     = 26.0,   -- px; close enough to fire to learn it hurts
  hitRadius      = 9.0,    -- px; close enough to a shot to count as hit
  harmCooldown   = 20,     -- frames before she can be hurt again

  -- Where the boundary between the small-object cells and the looming cells
  -- sits.  Below smallDeg of angular size, and expanding slower than
  -- expansionDeg per frame, a thing is not an escape stimulus at all.
  smallDeg       = 6.0,
  loomDeg        = 18.0,
  expansionDeg   = 0.40,
}

---------------------------------------------------------------------------
-- what she has learned.  One fly, one bar, kept for the run.
---------------------------------------------------------------------------
local valence = Valence.new()

-- Pain has to reach her through her eyes, not through a scripted knockback:
-- when something hurts her, it is presented as a maximal looming stimulus from
-- the direction it came from, and the escape that follows is computed by the
-- connectome like any other.  This holds that bearing for a few frames.
local pain = { az = 0.0, frames = 0 }

local function hurt(kind, key, azimuth)
  Valence.punish(valence, kind, key)
  pain.az = azimuth or 0.0
  pain.frames = 12
end

local function saveValence()
  local ok, json = pcall(require, "json")
  if ok and json then mod:SaveData(json.encode(Valence.serialise(valence))) end
end

local function loadValence()
  if not mod:HasData() then return end
  local ok, json = pcall(require, "json")
  if not ok or not json then return end
  local decoded
  if pcall(function() decoded = json.decode(mod:LoadData()) end) then
    Valence.restore(valence, decoded)
  end
end

-- Enemies the fly hurt, so a kill can be credited to her rather than to Isaac.
local damagedByFly = {}

-- A coarse name for a thing, so she can remember this *kind* of danger rather
-- than this one instance of it.
local function memoryKey(ent)
  if not ent then return nil end
  return tostring(ent.Type) .. "." .. tostring(ent.Variant)
end

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

    -- How fast *he* is coming, not how fast the two are converging.  Using the
    -- relative velocity made the fly her own threat: flying towards a
    -- motionless player counted as the player rushing her, and fear sat at
    -- 13.8 Hz instead of 1.2 (tools/diagnose_idle_fear.lua).  An animal that
    -- chooses to approach something is not being attacked by it -- which is
    -- what the fly's own optic-flow channel is for, telling self-motion apart
    -- from the world's.
    local pv = player.Velocity
    local closing = -(pv.X * dp.X + pv.Y * dp.Y) / pdist
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
                                  closing = 3.0, weight = 1.5,
                                  learned = Valence.threatScale(valence, "fire") }
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
        local key = memoryKey(ent)

        -- An enemy is also something to chase, on the same LC9/LC10 cells a
        -- fly tracks another fly with.  How much it is worth there is her
        -- nerve, so a frightened fly hardly notices and a bold one hunts.
        if ent:IsVulnerableEnemy() then
          tracked[#tracked + 1] = { az = az, angSize = angSize,
                                    weight = 1.0, prey = true }
        end
        if isObject then
          objects[#objects + 1] = { az = az, angSize = angSize }
        else
          -- How fast the object itself is coming, along the line of sight.
          -- The fly's own velocity is deliberately not in here: see the note
          -- on the player above.  A large enemy standing still is still
          -- frightening, but through its angular size, not through her flight.
          local v = ent.Velocity
          local closing = 0.0
          if v then
            closing = -(v.X * d.X + v.Y * d.Y) / dist
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
                                      weight = weight * loomness,
                                      prey = ent:IsVulnerableEnemy(),
                                      learned = Valence.threatScale(valence, key) }
          end
          if loomness < 0.98 then
            objects[#objects + 1] = { az = az,
                                      angSize = angSize * (1.0 - loomness) }
          end
        end
      end
    end
  end

  -- Walls loom too, and this is the one place the fly's own movement is the
  -- stimulus: flying at a surface really does make it expand, and that is how
  -- a fly avoids obstacles.  Previously this was a single yes/no test at a
  -- fixed distance, which slammed the looming channel with 95 Hz the frame a
  -- probe crossed the boundary -- a +22.9 Hz jump in fear while nothing in the
  -- room had moved.  Now each probe measures how far the wall actually is, and
  -- the drive follows from that distance and her speed towards it, so it is
  -- continuous and falls to nothing when she is not approaching.
  local speed = sqrt(fly.Velocity.X ^ 2 + fly.Velocity.Y ^ 2)
  for _, probe in ipairs(CFG.wallProbes) do
    local a = agent.heading + probe / DEG
    local ca, sa = cos(a), sin(a)
    local hit = nil
    local stride = CFG.wallRange / CFG.wallSteps
    for step = 1, CFG.wallSteps do
      local d = step * stride
      if not room:IsPositionInRoom(pos + Vector(ca * d, sa * d), 0) then
        hit = d
        break
      end
    end
    if hit then
      -- her speed along this probe is what makes the surface expand
      local closing = fly.Velocity.X * ca + fly.Velocity.Y * sa
      if closing < 0.0 then closing = 0.0 end
      walls[#walls + 1] = { az = probe, dist = hit, closing = closing }
    end
  end

  -- Whatever just hurt her, presented as the biggest thing she can see coming
  -- straight at her.  The flight that follows is the connectome's, not mine.
  if pain.frames > 0 then
    threats[#threats + 1] = { az = pain.az, angSize = 120.0,
                              closing = 14.0, weight = 2.2 }
    pain.frames = pain.frames - 1
  end

  return { tracked = tracked, threats = threats,
           objects = objects, walls = walls }
end

---------------------------------------------------------------------------
-- movement + contact damage
---------------------------------------------------------------------------
-- The enemy she is currently on, if any.
local function preyUnderFly(fly, reach)
  local best, bestD = nil, reach
  for _, ent in ipairs(Isaac.GetRoomEntities()) do
    if ent:IsVulnerableEnemy() then
      local d = ent.Position:Distance(fly.Position) - ent.Size
      if d < bestD then best, bestD = ent, d end
    end
  end
  return best
end

local function applyMovement(fly, agent, player)
  local vx, vy = agent.vx, agent.vy

  -- Riding.  Deciding to stay on the thing is the agent's -- see the feeding
  -- rule in agent.lua.  Actually going where it goes is not a decision at all,
  -- it is what being attached to a moving animal means, so it is done here.
  if agent.feeding then
    local prey = preyUnderFly(fly, CFG.feedReach)
    if prey then
      local d = prey.Position - fly.Position
      vx = prey.Velocity.X + d.X * CFG.feedCling
      vy = prey.Velocity.Y + d.Y * CFG.feedCling
      fly.Velocity = Vector(vx, vy)
      local sprite = fly:GetSprite()
      sprite.Rotation = 0.0
      if not sprite:IsPlaying("Idle") then sprite:Play("Idle", false) end
      return
    end
  end

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
  -- She is drawn upright and stays upright, like the game's own flies.  The
  -- heading the connectome computes still decides where she goes -- it just
  -- does not turn the picture any more, so the sprite reads the same at every
  -- bearing instead of lying on its side.
  sprite.Rotation = 0.0
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
        damagedByFly[key] = Game():GetFrameCount()
        -- she hit it and flew away intact: that is what courage is made of
        Valence.reward(valence, "hit", memoryKey(ent))
      end
    end
  end
end

---------------------------------------------------------------------------
-- what hurts her
---------------------------------------------------------------------------
-- Isaac familiars cannot be damaged, so being hurt has to be noticed rather
-- than received.  Fire she has flown into, and projectiles that reach her,
-- both count -- and both go in through the looming channel so the reaction is
-- the connectome's.
local function checkHarm(fly, agent)
  local pos = fly.Position
  local hx, hy = cos(agent.heading), sin(agent.heading)

  for _, item in ipairs(scanGrid()) do
    if item.kind == "fire" then
      local d = item.pos - pos
      if d:Length() < CFG.burnRadius then
        hurt("burn", "fire", relAzimuth(hx, hy, d.X, d.Y))
        return
      end
    end
  end

  for _, ent in ipairs(Isaac.GetRoomEntities()) do
    if ent.Type == EntityType.ENTITY_FIREPLACE then
      local d = ent.Position - pos
      if d:Length() < CFG.burnRadius + ent.Size then
        hurt("burn", "fire", relAzimuth(hx, hy, d.X, d.Y))
        return
      end
    elseif ent.Type == EntityType.ENTITY_PROJECTILE then
      local d = ent.Position - pos
      if d:Length() < CFG.hitRadius + ent.Size then
        hurt("shot", memoryKey(ent.SpawnerEntity), relAzimuth(hx, hy, d.X, d.Y))
        return
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
  -- She has to be able to sit *on* an enemy to feed on it.  With a collision
  -- class the engine shoves her back out the instant she touches one, which
  -- looks exactly like a fly that circles the room and never lands a hit --
  -- and it is invisible to the offline harness, which has no physics.
  fly.EntityCollisionClass = EntityCollisionClass.ENTCOLL_NONE
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

  -- What she has learned does not touch the network or the encoding: it moves
  -- three gains that decide which of two channels the connectome already
  -- computes gets to win.  See fly_brain/valence.lua.
  Valence.decay(valence)
  for k, v in pairs(Valence.gains(valence)) do agent.cfg[k] = v end

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

  -- Diagnostics: one line a second while a fly is alive, so a report of "she
  -- is not attacking" arrives with the numbers that would explain it.
  if TEST_MODE and Game():GetFrameCount() % 30 == 0 then
    local g = Valence.gains(valence)
    local nPrey, nearest = 0, 9999
    for _, t in ipairs(percepts.tracked or {}) do
      if t.prey then
        nPrey = nPrey + 1
        if t.angSize > 0 then nearest = math.min(nearest, t.angSize) end
      end
    end
    Isaac.DebugString(string.format(
      "[Connectome] courage %+.2f  hunt %.2f  threat %.2f  panic %.0f | "
      .. "prey seen %d  huntSize %.0f deg  huntAz %+.0f | "
      .. "escape %.1f Hz  feeding %s  land %.2f  speed %.1f",
      valence.courage, g.huntWeight, g.preyThreat, g.escapeThreshold,
      nPrey, agent.huntSize or 0, agent.huntAz or 0,
      agent.out and agent.out.escape or 0,
      tostring(agent.feeding), agent.landing or 0, agent.speed))
  end

  applyMovement(fly, agent, player)
  damageContact(fly, data, agent, player)

  data.harmCooldown = (data.harmCooldown or 0) - 1
  if data.harmCooldown <= 0 then
    local before = valence.courage
    checkHarm(fly, agent)
    if valence.courage ~= before then data.harmCooldown = CFG.harmCooldown end
  end
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
local C_BOLD = { 0.45, 0.90, 0.55 }

-- The bar runs terrified -> bold, so it needs both ends coloured, and it is
-- the thing worth looking at: everything below it is what the bar is doing.
local function renderCourage(x, y, w)
  local c = valence.courage
  Hud.stext(Valence.mood(valence), x, y, 0.5,
    c >= 0 and C_BOLD or C_FEAR)
  Hud.stext(string.format("%+.2f", c), x + w - 20, y, 0.45, C_DIM)
  Hud.thinBiBar(x, y + 8, w, 5, c, C_FEAR, C_BOLD)
  -- a flash on the frame something taught her
  if valence.lastEventAge < 14 then
    local up = valence.lastEvent == "hit" or valence.lastEvent == "kill"
    Hud.stext(up and "+" or "-", x + w * (0.5 + c * 0.5) - 2, y + 13, 0.5,
      up and C_BOLD or C_FEAR)
  end
end

local function renderCompact(agent)
  local o = agent.out
  local w = 104
  local x = Isaac.GetScreenWidth() - w - 14
  local y = Isaac.GetScreenHeight() - 52

  renderCourage(x, y, w)
  y = y + 18

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
  elseif agent.feeding then
    Hud.stext("ON IT", x + w - 22, y + 29, 0.45, C_BOLD)
  end
end

---------------------------------------------------------------------------
-- full read-out: everything, for tuning and for watching it think
---------------------------------------------------------------------------
local function renderFull(agent, fly)
  local x, y = 24, 36
  local bx, bw, bh = x + 78, 150, 9
  local o, d = agent.out, agent.drive
  local h = historyFor(GetPtrHash(fly))
  local active = agent.brain:activeCount()
  h.active:push(active)
  h.escape:push(o.escape)
  h.danger:push(d.loom)

  Hud.rect(x - 8, y - 8, 300, 242, 0.04, 0.05, 0.07, 0.72)
  Hud.text(string.format("%d neurons / %d synapses", net.n, net.nnz),
    x, y, C_LABEL[1], C_LABEL[2], C_LABEL[3])
  y = y + 14

  local g = Valence.gains(valence)
  Hud.text("LEARNED", x, y, 0.55, 0.8, 0.9)
  y = y + 13
  Hud.biBar(bx, y, bw, bh, valence.courage, "fear <-> courage",
    C_FEAR, C_BOLD, Valence.mood(valence))
  y = y + 13
  Hud.text(string.format("  hunt %.2f   threat %.2f   panic at %.0f Hz%s",
    g.huntWeight, g.preyThreat, g.escapeThreshold,
    agent.feeding and "   RIDING" or ""), x, y, 0.5, 0.6, 0.65)
  y = y + 17

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

-- Anything the fly hurts is remembered, so the kill is credited to her and not
-- to Isaac standing nearby.
function mod:onEntityKill(entity)
  local key = GetPtrHash(entity)
  if damagedByFly[key] then
    damagedByFly[key] = nil
    Valence.reward(valence, "kill", memoryKey(entity))
    saveValence()
    return
  end

  -- Something she was afraid of died where she could see it, and she is still
  -- here.  Worth less than doing it herself, but it is the only way a
  -- frightened fly ever gets out of the hole -- see fly_brain/valence.lua.
  --
  -- This callback fires for everything that dies, tears and effects included,
  -- so it has to be an actual hostile creature.  IsVulnerableEnemy is already
  -- false by now -- it is dead -- so IsEnemy is the test that still holds.
  if not (entity:ToNPC() and entity:IsEnemy()) then return end

  for _, fly in ipairs(Isaac.FindByType(EntityType.ENTITY_FAMILIAR, FLY_VARIANT, -1)) do
    if entity.Position:Distance(fly.Position) < CFG.visionRange then
      Valence.reward(valence, "witness", memoryKey(entity))
      saveValence()
      return
    end
  end
end

function mod:onNewRoom()
  damagedByFly = {}
  pain.frames = 0
end

function mod:onGameStarted(isContinued)
  if isContinued then
    loadValence()
  else
    valence = Valence.new()
    saveValence()
  end
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
mod:AddCallback(ModCallbacks.MC_POST_ENTITY_KILL, mod.onEntityKill)
mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, mod.onNewRoom)
mod:AddCallback(ModCallbacks.MC_PRE_GAME_EXIT, saveValence)
mod:AddCallback(ModCallbacks.MC_EVALUATE_CACHE, mod.onCache)
mod:AddCallback(ModCallbacks.MC_FAMILIAR_INIT, mod.onFamiliarInit, FLY_VARIANT)
mod:AddCallback(ModCallbacks.MC_FAMILIAR_UPDATE, mod.onFamiliarUpdate, FLY_VARIANT)
mod:AddCallback(ModCallbacks.MC_EXECUTE_CMD, mod.onCommand)
mod:AddCallback(ModCallbacks.MC_POST_RENDER, mod.onRender)

Isaac.DebugString("[Connectome] registered (item " .. tostring(ITEM) ..
  ", familiar variant " .. tostring(FLY_VARIANT) ..
  ", network " .. (net and "ready" or ("FAILED: " .. tostring(loadError))) .. ")")
