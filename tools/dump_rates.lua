-- Run the Lua simulator through the shared protocols and write per-neuron
-- firing rates to build/lua_rates.csv for comparison against the numpy
-- reference.  Run: tools/lua53/lua53.exe tools/dump_rates.lua [trials]
package.path = "mod/?.lua;mod/fly_brain/?.lua;" .. package.path

local Loader = require("fly_brain.loader")
local Brain = require("fly_brain.brain")
local packed = require("fly_brain.net_csr")
local meta = require("fly_brain.net_meta")
local net = Loader.load(packed, require)
net.vrest = require("fly_brain.net_vrest")

local TRIALS = tonumber(arg and arg[1]) or 8

-- Must stay identical to pipeline/protocols.py
local PROTOCOLS = {
  { "loom_right",     { loom = true, object = true },  45.0,  35.0, 150.0, 1000 },
  { "loom_left",      { loom = true, object = true }, -45.0,  35.0, 150.0, 1000 },
  { "loom_front",     { loom = true, object = true },   0.0,  30.0, 150.0, 1000 },
  { "landmark_front", { landmark = true },              0.0,  25.0, 120.0, 1000 },
  { "landmark_right", { landmark = true },             60.0,  25.0, 120.0, 1000 },
  { "flow_yaw",       { flow = true },                  0.0, 200.0,  80.0, 1000 },
}

local b = Brain.new(net, 1.0)
local out = assert(io.open("build/lua_rates.csv", "w"))
out:write("protocol,neuron,rate_hz\n")

for _, p in ipairs(PROTOCOLS) do
  local name, kinds, centre, sigma, peak, ms = p[1], p[2], p[3], p[4], p[5], p[6]
  local acc = {}
  for i = 1, net.n do acc[i] = 0 end

  for trial = 1, TRIALS do
    b:reset()
    b:seed(1000 + trial * 7919)
    for kind, _ in pairs(kinds) do
      local list = meta.sensory[kind]
      for k = 1, #list do
        local e = list[k]
        local d = (e[2] - centre) / sigma
        local hz = peak * math.exp(-0.5 * d * d)
        if hz > 1.0 then b:setInput(e[1], hz) end
      end
    end
    b:run(ms)
    for i = 1, net.n do acc[i] = acc[i] + b.spikeCount[i] end
  end

  local active = 0
  for i = 1, net.n do
    local hz = acc[i] / TRIALS / (ms / 1000.0)
    if hz > 0 then
      active = active + 1
      out:write(string.format("%s,%d,%.4f\n", name, i, hz))
    end
  end
  print(string.format("%-16s %d trials, %d neurons active", name, TRIALS, active))
end

out:close()
print("wrote build/lua_rates.csv")
