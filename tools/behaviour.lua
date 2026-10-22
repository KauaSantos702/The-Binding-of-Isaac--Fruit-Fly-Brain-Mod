--[[ behaviour.lua -- does the fly behave like a fly?

  A headless arena that runs the exact agent code the mod runs, so the
  behaviour can be checked without launching Isaac.  Three experiments, each
  the in-game version of a classic fly assay:

    1. looming avoidance  a threat approaches from a fixed bearing; a fly
                          turns away from it and speeds up.
    2. bearing sweep      the same threat presented at every bearing; the
                          turn response should be odd-symmetric -- left
                          threats turn the fly right and vice versa.
    3. free flight        no threat, just the landmark; the fly should keep
                          moving and stay in the arena rather than freeze or
                          spiral.

  Run: tools/lua53/lua53.exe tools/behaviour.lua
]]
package.path = "mod/?.lua;mod/fly_brain/?.lua;" .. package.path

local Loader  = require("fly_brain.loader")
local Brain   = require("fly_brain.brain")
local Sensory = require("fly_brain.sensory")
local Motor   = require("fly_brain.motor")
local Agent   = require("fly_brain.agent")

local net = Loader.load(require("fly_brain.net_csr"), require)
net.vrest = require("fly_brain.net_vrest")
local meta = require("fly_brain.net_meta")

local DEG = 180.0 / math.pi
local cos, sin, sqrt, atan, abs = math.cos, math.sin, math.sqrt, math.atan, math.abs

local function relAz(heading, dx, dy)
  local hx, hy = cos(heading), sin(heading)
  return atan(hx * dy - hy * dx, hx * dx + hy * dy) * DEG
end

local function wrap(a)
  while a > 180 do a = a - 360 end
  while a < -180 do a = a + 360 end
  return a
end

---------------------------------------------------------------------------
-- 1 + 2. a threat closes in from a given bearing
---------------------------------------------------------------------------
local function loomingTrial(bearingDeg, frames, seed)
  local agent = Agent.new(Brain, Sensory, Motor, net, meta, nil, seed)
  agent.heading = 0.0
  local fx, fy = 0.0, 0.0          -- fly position
  local px, py = 0.0, -120.0       -- landmark (the player)

  -- the threat starts 220 px away at `bearing` and closes at 4 px/frame
  local br = bearingDeg / DEG
  local dist = 220.0
  local turnAccum, escapeFrames, peakEscape = 0.0, 0, 0.0

  for f = 1, frames do
    local tx = fx + cos(agent.heading + br) * dist
    local ty = fy + sin(agent.heading + br) * dist
    local dx, dy = tx - fx, ty - fy
    local d = sqrt(dx * dx + dy * dy)
    local angSize = 2.0 * atan(14.0 / math.max(d, 8.0)) * DEG

    local percepts = {
      landmarkAz = relAz(agent.heading, px - fx, py - fy),
      threats = { { az = relAz(agent.heading, dx, dy), angSize = angSize,
                    closing = 4.0, weight = 1.0 } },
      objects = {}, walls = {},
    }
    local before = agent.heading
    agent:update(percepts)
    turnAccum = turnAccum + wrap((agent.heading - before) * DEG)
    if agent.escaping then escapeFrames = escapeFrames + 1 end
    if agent.out.escape > peakEscape then peakEscape = agent.out.escape end

    fx = fx + agent.vx
    fy = fy + agent.vy
    dist = math.max(26.0, dist - 4.0)
  end
  return turnAccum, escapeFrames, peakEscape, agent
end

print("=== 1. threat closing from the fly's right (+70 deg) ===")
local turn, esc, peak = loomingTrial(70, 60, 4242)
print(string.format("  net turn over 60 frames: %+.1f deg   (negative = away, to the left)", turn))
print(string.format("  escape DN peak %.1f Hz, escaping on %d/60 frames", peak, esc))

print("\n=== 2. turn response versus threat bearing ===")
print("  bearing   net turn   escape peak")
local sweep = {}
for _, b in ipairs({ -140, -100, -70, -40, -15, 15, 40, 70, 100, 140 }) do
  local t, e, p = loomingTrial(b, 45, 1000 + b)
  sweep[#sweep + 1] = { b = b, t = t, p = p }
  print(string.format("  %+5d deg  %+8.1f    %6.1f Hz", b, t, p))
end

-- odd symmetry: a threat on the left should turn the fly the other way from
-- the same threat on the right
local asym, n = 0.0, 0
for i = 1, #sweep // 2 do
  local l, r = sweep[i], sweep[#sweep + 1 - i]
  asym = asym + (l.t - r.t)
  n = n + 1
end
print(string.format("  mean (left turn - right turn) = %+.1f deg over %d mirrored pairs", asym / n, n))
print("  -> positive means the fly consistently turns away from the threat")

print("\n=== 3. free flight inside a room ===")
-- An Isaac room is roughly 540 x 300 px of walkable floor.  Walls loom, and in
-- an enclosed arena that is what actually keeps a fly turning.
local HALF_W, HALF_H = 270.0, 150.0

-- Wall sensing, identical to mod/main.lua: each probe measures how far the
-- surface actually is, and the drive follows from that distance and her speed
-- towards it.  The old version was a yes/no test at a fixed 46 px, which made
-- fear jump 22.9 Hz the frame a probe crossed the boundary.
local WALL_PROBES = { 0, 35, -35, 70, -70 }
local WALL_RANGE, WALL_STEPS = 110.0, 8

local function wallPercepts(a, fx, fy)
  local out = {}
  for _, probe in ipairs(WALL_PROBES) do
    local ang = a.heading + probe / DEG
    local ca, sa = cos(ang), sin(ang)
    local hit
    for step = 1, WALL_STEPS do
      local d = step * (WALL_RANGE / WALL_STEPS)
      if abs(fx + ca * d) > HALF_W or abs(fy + sa * d) > HALF_H then
        hit = d
        break
      end
    end
    if hit then
      local closing = (a.vx or 0) * ca + (a.vy or 0) * sa
      if closing < 0 then closing = 0 end
      out[#out + 1] = { az = probe, dist = hit, closing = closing }
    end
  end
  return out
end
-- Averaged over several seeds: the simulation is stochastic, and one seed is a
-- lottery rather than a measurement.
local function freeFlight(seed, frames)
  local a = Agent.new(Brain, Sensory, Motor, net, meta, nil, seed)
  a.heading = 0.4
  local fx, fy = 60.0, 0.0
  local px, py = 0.0, 0.0
  local pathLen, escaped, turns, totalTurn = 0.0, 0, 0, 0.0
  local seen, cells, prev = {}, 0, a.heading
  for _ = 1, frames do
    a:update({ landmarkAz = relAz(a.heading, px - fx, py - fy),
               threats = {}, objects = {},
               walls = wallPercepts(a, fx, fy) })
    fx = fx + a.vx
    fy = fy + a.vy
    if fx > HALF_W then fx = HALF_W elseif fx < -HALF_W then fx = -HALF_W end
    if fy > HALF_H then fy = HALF_H elseif fy < -HALF_H then fy = -HALF_H end
    pathLen = pathLen + sqrt(a.vx ^ 2 + a.vy ^ 2)
    local dt = abs(wrap((a.heading - prev) * DEG))
    if dt > 0.5 then turns = turns + 1 end
    totalTurn = totalTurn + dt
    prev = a.heading
    if a.escaping then escaped = escaped + 1 end
    local cx = math.floor((fx + HALF_W) / (2 * HALF_W) * 6)
    local cy = math.floor((fy + HALF_H) / (2 * HALF_H) * 4)
    if cx > 5 then cx = 5 end
    if cy > 3 then cy = 3 end
    local key = cx * 4 + cy
    if not seen[key] then seen[key] = true; cells = cells + 1 end
  end
  return { cells = cells, path = pathLen, turns = turns, turn = totalTurn,
           escaped = escaped }
end

-- Coverage replaces a turn-count check that was calibrated against the old
-- wall sensing.  That version slammed 95 Hz into the looming channel whenever
-- a probe crossed the boundary, and the constant shoving produced 147 turning
-- frames out of 239.  With a graded wall she flies in longer straight bouts
-- punctuated by saccades, which is what a real fly does -- so what to check is
-- whether she gets around the room, not how often she is knocked off course.
local FRAMES, SEEDS = 240, { 777, 1234, 4242, 9001 }
local cells, pathLen, turns, totalTurn, escaped = 0, 0.0, 0, 0.0, 0
for _, sd in ipairs(SEEDS) do
  local r = freeFlight(sd, FRAMES)
  cells = cells + r.cells / #SEEDS
  pathLen = pathLen + r.path / #SEEDS
  turns = turns + r.turns / #SEEDS
  totalTurn = totalTurn + r.turn / #SEEDS
  escaped = escaped + r.escaped / #SEEDS
end
print(string.format("  mean of %d seeds, %d frames each", #SEEDS, FRAMES))
print(string.format("  path length %.0f px (%.2f px/frame)", pathLen, pathLen / FRAMES))
print(string.format("  frames with a real turn: %.0f/%d, total turning %.0f deg",
  turns, FRAMES - 1, totalTurn))
print(string.format("  visited %.1f of the room's 24 cells", cells))
print(string.format("  wall-driven escape on %.0f frames", escaped))

print("\n=== 4. is the central complex doing anything? ===")
local function cxProbe(label, percepts, frames)
  local a = Agent.new(Brain, Sensory, Motor, net, meta, nil, 31337)
  for _ = 1, frames do a:update(percepts) end
  local function rate(name)
    local ids = meta.watch[name]
    if not ids then return -1 end
    return a.brain:popRate(ids, a.steps)
  end
  print(string.format("  %-22s EPG %5.1f  PEN1 %5.1f  D7 %5.1f  ER1 %5.1f  PFL3 %5.1f  PFL2 %5.1f Hz",
    label, rate("EPG"), rate("PEN_a(PEN1)"), rate("Delta7"), rate("ER1"),
    rate("PFL3"), rate("PFL2")))
end
cxProbe("landmark in front", { landmarkAz = 0, threats = {}, objects = {}, walls = {} }, 40)
cxProbe("landmark to the right", { landmarkAz = 75, threats = {}, objects = {}, walls = {} }, 40)
cxProbe("landmark behind", { landmarkAz = 170, threats = {}, objects = {}, walls = {} }, 40)
cxProbe("nothing at all", { threats = {}, objects = {}, walls = {} }, 40)

print("\n=== 5. does the circuit ever turn the fly TOWARD something? ===")
-- Escape is proven: the fly turns away from a looming threat.  Approach is a
-- different computation and must not be claimed without evidence.  Here one
-- silent, motionless object sits at a fixed bearing, presented only on the
-- object-position channel (LC9/LC10/LC11), and we ask which way the fly turns.
local function approachTrial(bearingDeg, kind, frames, seed)
  local a = Agent.new(Brain, Sensory, Motor, net, meta, nil, seed)
  a.heading = 0.0
  local turnSum = 0.0
  for _ = 1, frames do
    local p = { threats = {}, objects = {}, walls = {} }
    if kind == "object" then
      p.objects = { { az = bearingDeg, angSize = 16 } }
    else
      p.landmarkAz = bearingDeg
    end
    local before = a.heading
    a:update(p)
    turnSum = turnSum + wrap((a.heading - before) * DEG)
  end
  return turnSum
end

print("  bearing   as an object   as the landmark   (+ = turned right)")
local toward = 0
for _, b in ipairs({ -110, -60, -25, 25, 60, 110 }) do
  local o = approachTrial(b, "object", 40, 606 + b)
  local l = approachTrial(b, "landmark", 40, 707 + b)
  print(string.format("  %+5d deg   %+8.1f       %+8.1f", b, o, l))
  if (b > 0 and o > 0) or (b < 0 and o < 0) then toward = toward + 1 end
end
print(string.format("  turned toward the object in %d of 6 bearings", toward))


print("\n=== 6. Isaac: still versus moving ===")
-- The player is on the tracking channel always, and on the looming channel
-- only in proportion to how fast he moves -- the same rule main.lua uses.
-- A fly should drift toward a motionless animal and bolt when it lunges.
local function isaacTrial(playerSpeed, frames, seed)
  local a = Agent.new(Brain, Sensory, Motor, net, meta, nil, seed)
  a.heading = 0.0
  local fx, fy = 0.0, 0.0
  local px, py = 150.0, 0.0        -- Isaac, 150 px to the fly's right
  local startDist, endDist, fled, peakFear = 0.0, 0.0, 0, 0.0
  local speedSum, finalSpeed = 0.0, 0.0

  for f = 1, frames do
    -- Isaac charges straight at the fly when playerSpeed > 0
    local dx, dy = fx - px, fy - py
    local dl = math.max(sqrt(dx * dx + dy * dy), 1.0)
    px = px + dx / dl * playerSpeed
    py = py + dy / dl * playerSpeed

    local ox, oy = px - fx, py - fy
    local dist = sqrt(ox * ox + oy * oy)
    if f == 1 then startDist = dist end
    local size = 2.0 * atan(14.0 / math.max(dist, 10.0)) * DEG
    local az = relAz(a.heading, ox, oy)

    local p = { tracked = { { az = az, angSize = size, weight = 1.0 } },
                threats = {}, objects = {}, walls = {} }
    -- closing speed plus raw motion, as in main.lua
    local startle = playerSpeed + 0.35 * playerSpeed
    if startle > 0.4 then
      p.threats[1] = { az = az, angSize = size, closing = startle,
                       weight = 0.9, sizeGain = 0.0 }
    end

    a:update(p)
    if a.escaping then fled = fled + 1 end
    if a.out.escape > peakFear then peakFear = a.out.escape end
    speedSum = speedSum + sqrt(a.vx * a.vx + a.vy * a.vy)
    if f > frames - 15 then finalSpeed = finalSpeed + sqrt(a.vx * a.vx + a.vy * a.vy) / 15 end
    fx = fx + a.vx
    fy = fy + a.vy
    endDist = sqrt((px - fx) ^ 2 + (py - fy) ^ 2)
  end
  return startDist, endDist, fled, peakFear, speedSum / frames, finalSpeed
end

print("  Isaac speed   distance start -> end   frames fleeing   peak fear   her speed")
local stillEnd, movingEnd
for _, sp in ipairs({ 0.0, 1.5, 5.0 }) do
  local s, e, fled, fear, spd, endSpd = isaacTrial(sp, 70, 4321)
  if sp == 0.0 then stillEnd = e end
  if sp == 5.0 then movingEnd = e end
  print(string.format("  %5.1f px/f      %5.0f -> %5.0f px          %3d/70        %5.1f Hz    %4.1f px/f   settled at %4.2f px/f",
    sp, s, e, fled, fear, spd, endSpd))
end
print("  -> standing still should let her close the distance; charging should not")


print("\n=== 7. does size decide whether something is frightening? ===")
-- Isaac is full of fly enemies, and the question is whether the circuit treats
-- them as threats.  A real fly cannot recognise a conspecific: the separation
-- happens because small slow objects drive LC11/LC17/LC18 and not LC4/LPLC2.
-- So this presents things of different size at the same distance and speed,
-- routed exactly as main.lua routes them.
local SMALL_DEG, LOOM_DEG, EXPANSION_DEG = 6.0, 18.0, 0.40

local function route(radius, dist, closing)
  local angSize = 2.0 * atan(radius / dist) * DEG
  local bySize = (angSize - SMALL_DEG) / (LOOM_DEG - SMALL_DEG)
  local byExp = (angSize * closing / dist) / EXPANSION_DEG
  local loom = bySize > byExp and bySize or byExp
  if loom < 0 then loom = 0 elseif loom > 1 then loom = 1 end
  return angSize, loom
end

local function fearOf(radius, dist, closing, frames, seed)
  local a = Agent.new(Brain, Sensory, Motor, net, meta, nil, seed)
  a.heading = 0.0
  local peak = 0.0
  local angSize, loom = route(radius, dist, closing)
  for _ = 1, frames do
    local p = { tracked = {}, threats = {}, objects = {}, walls = {} }
    if loom > 0.02 then
      p.threats[1] = { az = 40, angSize = angSize, closing = closing, weight = loom }
    end
    if loom < 0.98 then
      p.objects[1] = { az = 40, angSize = angSize * (1 - loom) }
    end
    a:update(p)
    if a.out.escape > peak then peak = a.out.escape end
  end
  return angSize, loom, peak
end

print("  what                        size  dist  closing   angle  loom  peak fear")
local cases = {
  { "Attack Fly, far",          6.0, 130.0, 1.5 },
  { "Attack Fly, in her face",  6.0,  36.0, 1.5 },
  { "a tear, far but fast",     5.0, 130.0, 12.0 },
  { "Gaper, far",              13.0, 130.0, 1.5 },
  { "Gaper, close",            13.0,  50.0, 1.5 },
  { "Monstro, far",            22.0, 130.0, 1.0 },
}
local flyFar, gaperFar
for _, c in ipairs(cases) do
  local ang, loom, peak = fearOf(c[2], c[3], c[4], 40, 8080)
  if c[1] == "Attack Fly, far" then flyFar = peak end
  if c[1] == "Gaper, far" then gaperFar = peak end
  print(string.format("  %-26s %4.0f  %4.0f  %6.1f   %5.1f  %4.2f   %5.1f Hz",
    c[1], c[2], c[3], c[4], ang, loom, peak))
end


print("\n=== 8. fear while the player stands perfectly still ===")
-- The bug this catches: main.lua used to take the *relative* closing speed for
-- the player, so the fly flying towards a motionless Isaac counted as Isaac
-- rushing her, and the wall probes were a yes/no test that slammed 95 Hz into
-- the looming channel the frame one crossed the boundary.  Experiment 6 missed
-- both, because it modelled startle as a function of the player's speed alone
-- and never looked at fear in a room with walls.
local function idleTrial(label, startX, startY, frames)
  local a = Agent.new(Brain, Sensory, Motor, net, meta, nil, 2468)
  a.heading = math.pi
  local fx, fy = startX, startY
  local px, py = 0.0, 0.0          -- Isaac, not moving at all, ever
  -- Skip the first frames: the brain starts in total silence, so frame one is
  -- always a step from nothing to whatever is in view.  In game she has been
  -- running for hundreds of frames by the time anyone looks.
  local WARMUP = 12
  local peak, fled, jump, prev = 0.0, 0, 0.0, 0.0
  for frame = 1, frames do
    -- exactly the percepts main.lua builds for a room holding only the player
    local dx, dy = px - fx, py - fy
    local dist = sqrt(dx * dx + dy * dy)
    local p = { tracked = {}, threats = {}, objects = {},
                walls = wallPercepts(a, fx, fy) }
    if dist > 1.0 then
      local size = 2.0 * atan(14.0 / math.max(dist, 10.0)) * DEG
      p.tracked[1] = { az = relAz(a.heading, dx, dy), angSize = size, weight = 1.0 }
      -- the player is motionless, so there is no closing term at all
    end
    a:update(p)
    if frame > WARMUP then
      if a.out.escape > peak then peak = a.out.escape end
      if a.escaping then fled = fled + 1 end
      local dj = a.out.escape - prev
      if dj > jump then jump = dj end
    end
    prev = a.out.escape
    fx = fx + a.vx
    fy = fy + a.vy
    if fx > HALF_W then fx = HALF_W elseif fx < -HALF_W then fx = -HALF_W end
    if fy > HALF_H then fy = HALF_H elseif fy < -HALF_H then fy = -HALF_H end
  end
  print(string.format("  %-28s peak fear %5.1f Hz   fleeing %2d/%d   "
    .. "biggest jump %+5.1f Hz", label, peak, fled, frames, jump))
  return peak, fled, jump
end

local idleMidPeak, idleMidFled, idleMidJump = idleTrial("mid-room", 120, 0, 90)
local idleWallPeak, idleWallFled, idleWallJump = idleTrial("hugging a wall", 250, 130, 90)


print("\n=== 9. can she be taught? ===")
-- The claim is that experience moves one bar, that bar moves three gains, and
-- those gains decide which of two channels the connectome already computes
-- wins.  Nothing about the encoding changes, so the only way to see it is in
-- what she does with the same enemy.
local Valence = require("fly_brain.valence")

local function teach(events)
  local v = Valence.new()
  for _, e in ipairs(events) do
    if Valence.REWARD[e] then Valence.reward(v, e, "gaper")
    else Valence.punish(v, e, "fire") end
  end
  return v
end

local function encounter(v, bearing, frames, seed)
  local a = Agent.new(Brain, Sensory, Motor, net, meta, Valence.gains(v), seed)
  a.heading = 0.0
  local fx, fy = 0.0, 0.0
  local dist = 150.0
  local toward, fled, closest = 0.0, 0, 999.0

  for _ = 1, frames do
    local ex = fx + cos(bearing / DEG) * dist
    local ey = fy + sin(bearing / DEG) * dist
    local dx, dy = ex - fx, ey - fy
    local d = sqrt(dx * dx + dy * dy)
    local az = relAz(a.heading, dx, dy)
    local size = 2.0 * atan(13.0 / math.max(d, 10.0)) * DEG
    local before = a.heading
    a:update({
      tracked = { { az = az, angSize = size, weight = 1.0, prey = true } },
      threats = { { az = az, angSize = size, closing = 0.0,
                    weight = 1.0, prey = true } },
      objects = {}, walls = {},
    })
    local turned = wrap((a.heading - before) * DEG)
    toward = toward + (az >= 0 and turned or -turned)
    if a.escaping then fled = fled + 1 end
    fx = fx + a.vx
    fy = fy + a.vy
    d = sqrt((ex - fx) ^ 2 + (ey - fy) ^ 2)
    if d < closest then closest = d end
    dist = math.max(28.0, d)
  end
  return toward, fled, closest
end

local lessons = {
  { "untaught",            {} },
  { "burned once",         { "burn" } },
  { "burned twice",        { "burn", "burn" } },
  { "a few good hits",     { "hit", "hit", "hit", "hit", "hit", "hit" } },
  { "hits and a kill",     { "hit", "hit", "hit", "kill", "hit", "kill" } },
  { "a hunter",            { "kill", "kill", "kill", "kill", "kill" } },
  -- the way out of the trap: she never touched a thing, she only watched
  { "burned, then watched", { "burn", "burn", "witness", "witness", "witness",
                              "witness", "witness", "witness", "witness" } },
  { "only ever watched",   { "witness", "witness", "witness", "witness",
                             "witness", "witness" } },
}

print("  what she has been through   courage   mood        turn toward   "
  .. "closed to   fleeing")
local result = {}
for _, lesson in ipairs(lessons) do
  local v = teach(lesson[2])
  local tw, fl, cl = 0.0, 0, 0.0
  for _, b in ipairs({ -60, -25, 25, 60 }) do
    local t, f, c = encounter(v, b, 60, 5000 + b)
    tw = tw + t; fl = fl + f; cl = cl + c / 4
  end
  result[lesson[1]] = { courage = v.courage, toward = tw, fled = fl, close = cl }
  print(string.format("  %-26s %+7.2f   %-10s %+9.0f deg %9.0f px %8d/240",
    lesson[1], v.courage, Valence.mood(v), tw, cl, fl))
end

-- and what she remembers about one particular thing
local burned = teach({ "burn" })
print(string.format("\n  after one burn, fire reads %.2fx as threatening; "
  .. "a beaten gaper reads %.2fx",
  Valence.threatScale(burned, "fire"),
  Valence.threatScale(teach({ "hit", "hit", "kill" }), "gaper")))


print("\n=== 10. does a bold fly actually stay on the enemy? ===")
-- The shape asked for: close in, ride it for a few seconds dealing contact
-- damage, then let go and come back round.  Riding is the landing rule from
-- experiment 3 applied to prey; letting go is satiation, which the connectome
-- has no notion of and which is declared as ours in agent.lua.
local REACH, COOLDOWN, FEED_REACH, CLING = 22, 12, 26, 0.30

local function attackRun(courage, frames, seed)
  local v = Valence.new(); v.courage = courage
  local a = Agent.new(Brain, Sensory, Motor, net, meta, Valence.gains(v), seed)
  a.heading = 0.0
  local fx, fy = 0.0, 0.0
  local ex, ey, evx, evy = 150.0, 0.0, -0.7, 0.35
  local hits, cd, riding, bouts, was = 0, 0, 0, 0, false
  for _ = 1, frames do
    ex = ex + evx; ey = ey + evy
    if abs(ex) > 250 then evx = -evx end
    if abs(ey) > 140 then evy = -evy end
    local dx, dy = ex - fx, ey - fy
    local d = sqrt(dx * dx + dy * dy)
    local size = 2.0 * atan(13.0 / math.max(d, 10.0)) * DEG
    local az = relAz(a.heading, dx, dy)
    a:update({ tracked = { { az = az, angSize = size, weight = 1.0, prey = true } },
               threats = { { az = az, angSize = size, closing = 0.0,
                             weight = 1.0, prey = true } },
               objects = {}, walls = {} })
    if a.feeding and d < FEED_REACH + 13 then
      fx = fx + evx + (ex - fx) * CLING
      fy = fy + evy + (ey - fy) * CLING
      riding = riding + 1
      if not was then bouts = bouts + 1 end
      was = true
    else
      fx = fx + a.vx; fy = fy + a.vy
      was = false
    end
    d = sqrt((ex - fx) ^ 2 + (ey - fy) ^ 2)
    cd = cd - 1
    if d < REACH and cd <= 0 then hits = hits + 1; cd = COOLDOWN end
  end
  return hits, riding, bouts
end

print("  courage   hits/20s   frames riding   runs   seconds per run")
local cycle = {}
for _, c in ipairs({ 0.0, 0.3, 0.6, 1.0 }) do
  local h, r, b = 0, 0, 0
  for _, sd in ipairs({ 7, 17, 27, 37, 47 }) do
    local x, y, z = attackRun(c, 600, sd); h = h + x / 5; r = r + y / 5; b = b + z / 5
  end
  cycle[c] = { hits = h, riding = r, bouts = b }
  print(string.format("  %+5.1f %9.1f %13.0f/600 %6.1f %14s", c, h, r, b,
    b > 0 and string.format("%.1f", r / b / 30) or "-"))
end

print("\n=== checks ===")
local ok = true
local function check(cond, msg)
  print((cond and "  PASS  " or "  FAIL  ") .. msg)
  ok = ok and cond
end
check(turn < -2.0, "a threat on the right turns the fly left (away)")
check(peak > 25.0, "looming drives the escape descending neurons above 25 Hz")
check(asym / n > 2.0, "the turn response is odd-symmetric in threat bearing")
check(pathLen / 240 > 1.5, "the fly keeps flying rather than stalling")
check(cells >= 8 and turns > 25, "she gets around the room rather than stalling")
-- the tracking channel must pull her toward a target, not just away from threats
local towardL, towardR = approachTrial(-60, "landmark", 40, 111), approachTrial(60, "landmark", 40, 222)
check(towardL < 0 and towardR > 0, "LC9/LC10 turn the fly toward a tracked target")
local still = select(2, isaacTrial(0.0, 70, 4321))
local charge = select(4, isaacTrial(5.0, 70, 4321))
check(still < 60, "she closes on a motionless Isaac")
check(charge > 30, "a charging Isaac frightens her")
local _, _, _, _, _, settled = isaacTrial(0.0, 70, 4321)
check(settled < 1.0, "she settles on a motionless target instead of circling it")
check(flyFar < gaperFar * 0.5, "a distant fly-sized enemy frightens her far less than a big one")
check(idleMidFled == 0 and idleMidPeak < 10.0, "a motionless player does not frighten her")
check(idleWallFled == 0 and idleWallPeak < 22.0, "hugging a wall does not tip her into escape")
check(idleMidJump < 6.0 and idleWallJump < 6.0,
  "fear has no jump beyond read-out noise while nothing moves")
check(result["a hunter"].fled < result["burned twice"].fled,
  "a fly that has been winning runs less than one that has been burned")
check(result["a hunter"].toward > result["burned twice"].toward,
  "and turns towards the same enemy the burned one turns away from")
check(result["a hunter"].close < result["burned twice"].close,
  "and closes on it instead of keeping its distance")

-- The trap: frightened she will not approach, so she cannot land a hit, so she
-- stays frightened.  Watching Isaac win has to be able to pull her out of it
-- without her risking anything.
check(result["burned, then watched"].courage > result["burned twice"].courage,
  "watching a kill lifts a burned fly out of the fear it cannot escape alone")
check(result["burned, then watched"].fled <= result["untaught"].fled * 1.2,
  "a roomful of watched kills puts a scared fly back to about baseline")
check(result["only ever watched"].courage < result["a hunter"].courage,
  "but watching is still worth less than doing it herself")

check(cycle[1.0].hits > cycle[0.0].hits * 8 + 5,
  "a bold fly does real damage where a neutral one does almost none")
check(cycle[1.0].riding > 250,
  "she spends most of a fight actually on the enemy")
local perRun = cycle[1.0].riding / math.max(cycle[1.0].bouts, 1) / 30
check(perRun > 4.0 and perRun < 9.0,
  "each run lasts the few seconds it is meant to, then she lets go")
os.exit(ok and 0 or 1)
