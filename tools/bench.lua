-- Offline harness: load the connectome, check it decoded correctly, and time
-- the simulator at the rate Isaac will ask for (30 game ticks/s, dt = 1 ms,
-- so 33 brain steps per tick).   Run:  tools/lua53/lua53.exe tools/bench.lua
package.path = "mod/?.lua;mod/fly_brain/?.lua;" .. package.path

local Loader = require("fly_brain.loader")
local Brain = require("fly_brain.brain")

local t0 = os.clock()
local packed = require("fly_brain.net_csr")
local meta = require("fly_brain.net_meta")
local net = Loader.load(packed, require)
net.vrest = require("fly_brain.net_vrest")
local tLoad = os.clock() - t0

print(string.format("loaded %d neurons / %d synapses in %.2f s",
  net.n, net.nnz, tLoad))

-- structural checks -------------------------------------------------------
assert(#net.indptr == net.n + 1, "indptr length")
assert(net.indptr[1] == 1, "indptr must be 1-based")
assert(net.indptr[net.n + 1] == net.nnz + 1, "indptr tail")
assert(#net.indices == net.nnz, "indices length")
local minIdx, maxIdx, wsum, wmin, wmax = math.huge, -math.huge, 0, math.huge, -math.huge
for k = 1, net.nnz do
  local j = net.indices[k]
  if j < minIdx then minIdx = j end
  if j > maxIdx then maxIdx = j end
  local w = net.weights[k]
  wsum = wsum + math.abs(w)
  if w < wmin then wmin = w end
  if w > wmax then wmax = w end
end
print(string.format("  indices in [%d, %d], weights in [%.2f, %.2f] mV, |w| sum %.0f mV",
  minIdx, maxIdx, wmin, wmax, wsum))
assert(minIdx >= 1 and maxIdx <= net.n, "index out of range")

-- behaviour ---------------------------------------------------------------
local b = Brain.new(net, 1.0)
b:seed(12345)

-- 1. silence
b:reset()
b:run(500)
local spk = 0
for i = 1, net.n do spk = spk + b.spikeCount[i] end
print(string.format("quiet 500 ms: %d spikes, %d active", spk, b:activeCount()))
assert(spk == 0, "network must be silent with no input")

-- 2. looming on the right eye
b:reset()
local loom = meta.sensory.loom
local driven = 0
for k = 1, #loom do
  local e = loom[k]
  local i, az, _, side = e[1], e[2], e[3], e[4]
  if side == 1 then
    local d = (az - 45.0) / 35.0
    local hz = 150.0 * math.exp(-0.5 * d * d)
    if hz > 1 then b:setInput(i, hz); driven = driven + 1 end
  end
end
local t1 = os.clock()
b:run(500)
local tSim = os.clock() - t1

local function popRate(ids, ms)
  local s = 0
  for k = 1, #ids do s = s + b.spikeCount[ids[k]] end
  return s / #ids / (ms / 1000.0)
end
print(string.format("looming right (%d driven): %.1f ms for 500 steps -> %.3f ms/step",
  driven, tSim * 1000, tSim * 1000 / 500))
print(string.format("  DN left %.1f Hz   DN right %.1f Hz   escape %.1f Hz   active %d",
  popRate(meta.motor.dn_left, 500), popRate(meta.motor.dn_right, 500),
  popRate(meta.motor.escape, 500), b:activeCount()))

-- 3. frame budget: 33 steps per Isaac tick, sustained
b:reset()
for k = 1, #loom do
  local e = loom[k]
  if e[4] == 1 then
    local d = (e[2] - 45.0) / 35.0
    b:setInput(e[1], 150.0 * math.exp(-0.5 * d * d))
  end
end
b:run(200) -- warm up to steady state
local frames = 300
local worst, total = 0, 0
for _ = 1, frames do
  local s = os.clock()
  b:run(33)
  local el = os.clock() - s
  total = total + el
  if el > worst then worst = el end
end
print(string.format("frame budget: mean %.3f ms, worst %.3f ms per 33-step tick (%d active)",
  total / frames * 1000, worst * 1000, b:activeCount()))
print(string.format("  -> %.1f%% of a 33.3 ms Isaac tick", total / frames * 1000 / 33.3 * 100))

-- 4. the realistic in-game load: landmark + threats + walls, through the same
--    agent the mod uses.
local Sensory = require("fly_brain.sensory")
local Motor = require("fly_brain.motor")
local Agent = require("fly_brain.agent")

local function scenarioCost(label, percepts, cfg)
  local a = Agent.new(Brain, Sensory, Motor, net, meta, cfg, 99)
  for _ = 1, 40 do a:update(percepts) end          -- reach steady state
  local t = os.clock()
  local frames = 200
  for _ = 1, frames do a:update(percepts) end
  local ms = (os.clock() - t) / frames * 1000
  print(string.format("  %-34s %5.2f ms/frame (%4.1f%% of a tick), %4d active",
    label, ms, ms / 33.3 * 100, a.brain:activeCount()))
  return ms
end

print("agent-level cost (33 brain steps per frame):")
local threats = { { az = 55, angSize = 18, closing = 4, weight = 1 },
                  { az = -30, angSize = 12, closing = 2, weight = 1.4 } }
scenarioCost("idle, landmark only", { landmarkAz = 20, threats = {}, objects = {}, walls = {} })
scenarioCost("landmark + 2 threats + wall", { landmarkAz = 20, threats = threats,
  objects = { { az = -80, angSize = 9 } },
  walls = { { az = 0, dist = 40, closing = 2.6 } } })
scenarioCost("same, landmark drive halved", { landmarkAz = 20, threats = threats,
  objects = { { az = -80, angSize = 9 } },
  walls = { { az = 0, dist = 40, closing = 2.6 } } }, { landmarkPeak = 55 })
scenarioCost("same, landmark off", { threats = threats,
  objects = { { az = -80, angSize = 9 } },
  walls = { { az = 0, dist = 40, closing = 2.6 } } })
