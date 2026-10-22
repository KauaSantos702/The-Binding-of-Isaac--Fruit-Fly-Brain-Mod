-- Why does fear spike while the player stands still?
--
-- Analysis only: this does not touch the mod.  It rebuilds the percepts
-- exactly as mod/main.lua builds them -- including the two terms the
-- behaviour suite never exercised -- and logs fear frame by frame.
--
-- tools/behaviour.lua experiment 6 reported "0/70 frames fleeing" against a
-- motionless player, but it computed the startle term as
--     startle = playerSpeed + 0.35 * playerSpeed
-- which is zero when the player is still.  main.lua instead uses the *relative*
-- closing speed, which contains the fly's own velocity:
--     closing = ((flyVelocity - playerVelocity) . toPlayer) / distance
-- So the test validated a simplification, not the code.
--
-- Run: tools/lua53/lua53.exe tools/diagnose_idle_fear.lua
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
local cos, sin, sqrt, atan, abs, max = math.cos, math.sin, math.sqrt, math.atan,
                                       math.abs, math.max

-- constants copied from mod/main.lua CFG
local WALL_PROBES = { 0, 35, -35, 70, -70 }
local WALL_RANGE, WALL_STEPS = 110.0, 8
local HALF_W, HALF_H = 270.0, 150.0

local function relAz(heading, dx, dy)
  local hx, hy = cos(heading), sin(heading)
  return atan(hx * dy - hy * dx, hx * dx + hy * dy) * DEG
end

-- Exactly mod/main.lua's buildPercepts, for a room containing only the player.
-- `selfMotion` switches off the fly's own contribution to the closing term, so
-- the two can be compared.
local function percepts(agent, fx, fy, px, py, pvx, pvy, selfMotion, walls_on)
  local dx, dy = px - fx, py - fy
  local dist = sqrt(dx * dx + dy * dy)
  local p = { tracked = {}, threats = {}, objects = {}, walls = {} }

  if dist > 1.0 then
    local az = relAz(agent.heading, dx, dy)
    local size = 2.0 * atan(14.0 / max(dist, 10.0)) * DEG
    p.tracked[1] = { az = az, angSize = size, weight = 1.0 }

    local vx = selfMotion and agent.vx or 0.0
    local vy = selfMotion and agent.vy or 0.0
    local closing = -((pvx - vx) * dx + (pvy - vy) * dy) / dist
    if closing < 0.0 then closing = 0.0 end
    local pspeed = sqrt(pvx * pvx + pvy * pvy)
    local startle = closing + 0.35 * pspeed
    if startle > 0.4 then
      p.threats[1] = { az = az, angSize = size, closing = startle,
                       weight = 0.9, sizeGain = 0.0 }
    end
    p.closing = closing
    p.dist = dist
    p.size = size
  end

  if walls_on then
    for _, probe in ipairs(WALL_PROBES) do
      local a = agent.heading + probe / DEG
      local ca, sa = cos(a), sin(a)
      local blocked = function(d)
        return abs(fx + ca * d) > HALF_W or abs(fy + sa * d) > HALF_H
      end
      local hit
      for step = 1, WALL_STEPS do
        local d = step * (WALL_RANGE / WALL_STEPS)
        if blocked(d) then hit = d break end
      end
      if hit then
        local closing = (agent.vx or 0) * ca + (agent.vy or 0) * sa
        if closing < 0 then closing = 0 end
        p.walls[#p.walls + 1] = { az = probe, dist = hit, closing = closing }
      end
    end
  end
  return p
end

local function trial(label, selfMotion, walls_on, startX, startY, frames)
  local a = Agent.new(Brain, Sensory, Motor, net, meta, nil, 2468)
  a.heading = math.pi          -- pointing at the player, who is at the origin
  local fx, fy = startX, startY
  local px, py = 0.0, 0.0      -- the player, standing perfectly still
  local peak, fled, wallFrames = 0.0, 0, 0
  local maxJump, prev = 0.0, 0.0
  local trace = {}

  for f = 1, frames do
    local p = percepts(a, fx, fy, px, py, 0.0, 0.0, selfMotion, walls_on)
    a:update(p)
    local fear = a.out.escape
    if fear > peak then peak = fear end
    if a.escaping then fled = fled + 1 end
    if #p.walls > 0 then wallFrames = wallFrames + 1 end
    local jump = fear - prev
    if jump > maxJump then maxJump = jump end
    prev = fear
    trace[f] = { f = f, fear = fear, dist = p.dist or 0, loom = a.drive.loom,
                 walls = #p.walls, speed = sqrt(a.vx ^ 2 + a.vy ^ 2) }
    fx = fx + a.vx
    fy = fy + a.vy
    if fx > HALF_W then fx = HALF_W elseif fx < -HALF_W then fx = -HALF_W end
    if fy > HALF_H then fy = HALF_H elseif fy < -HALF_H then fy = -HALF_H end
  end

  print(string.format("  %-38s peak fear %5.1f Hz   fleeing %3d/%d   "
    .. "biggest 1-frame jump %+5.1f Hz", label, peak, fled, frames, maxJump))
  return trace, peak, fled, maxJump
end

print("The player never moves in any of these. The fly starts 120 px away.\n")

print("=== A. is she frightening herself? ===")
trial("as the mod does it (self-motion in)", true, false, 120, 0, 90)
trial("with her own velocity removed", false, false, 120, 0, 90)

print("\n=== B. what the wall probes add ===")
trial("mid-room, walls on", true, true, 120, 0, 90)
trial("starting near a wall, walls on", true, true, 250, 130, 90)
trial("starting near a wall, walls off", true, false, 250, 130, 90)

print("\n=== C2. frame by frame, hugging a wall (player term already fixed) ===")
do
  local a = Agent.new(Brain, Sensory, Motor, net, meta, nil, 2468)
  a.heading = math.pi
  local fx, fy = 250.0, 130.0
  local prev = 0.0
  print("  frame  walls  nearest  wall drive    fear    jump   speed")
  for f = 1, 60 do
    local p = percepts(a, fx, fy, 0, 0, 0, 0, false, true)
    a:update(p)
    local nearest = 999
    for _, w in ipairs(p.walls) do
      if w.dist < nearest then nearest = w.dist end
    end
    local jump = a.out.escape - prev
    if f % 2 == 0 then
      print(string.format("  %5d  %5d  %7.1f  %10.1f  %6.1f  %+6.1f  %5.2f",
        f, #p.walls, nearest, a.drive.loom, a.out.escape, jump,
        sqrt(a.vx ^ 2 + a.vy ^ 2)))
    end
    prev = a.out.escape
    fx = fx + a.vx
    fy = fy + a.vy
    if fx > HALF_W then fx = HALF_W elseif fx < -HALF_W then fx = -HALF_W end
    if fy > HALF_H then fy = HALF_H elseif fy < -HALF_H then fy = -HALF_H end
  end
end

print("\n=== C. frame by frame, approaching a motionless player ===")
local trace = trial("trace", true, false, 120, 0, 60)
print("  frame   distance   angular size   loom drive   fear    speed")
for _, r in ipairs(trace) do
  if r.f % 4 == 0 then
    local size = 2.0 * atan(14.0 / max(r.dist, 10.0)) * DEG
    print(string.format("  %5d   %7.1f   %10.1f    %9.1f  %6.1f   %5.2f",
      r.f, r.dist, size, r.loom, r.fear, r.speed))
  end
end

print("\n=== D. does the rate read-out itself jump when steps are throttled? ===")
-- popRate's window is dt * stepsElapsed.  The mod shrinks stepsPerFrame when a
-- frame runs over budget, which changes that window -- and with only 16 escape
-- neurons, one spike lands very differently at 33 steps than at 8.
local a = Agent.new(Brain, Sensory, Motor, net, meta, nil, 99)
a.heading = math.pi
for _ = 1, 30 do
  a:update(percepts(a, 60, 0, 0, 0, 0, 0, true, false))
end
for _, steps in ipairs({ 33, 24, 16, 8 }) do
  a.steps = steps
  local sum, n = 0.0, 12
  for _ = 1, n do
    a:update(percepts(a, 60, 0, 0, 0, 0, 0, true, false))
    sum = sum + a.out.escape
  end
  print(string.format("  %2d steps/frame -> %4.0f ms window, mean fear %5.1f Hz",
    steps, steps * 1.0, sum / n))
end
