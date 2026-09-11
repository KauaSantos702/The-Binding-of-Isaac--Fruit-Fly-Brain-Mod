--[[ brain.lua -- leaky integrate-and-fire simulator for a real Drosophila
     sub-connectome, small and allocation-free enough to run inside Isaac.

  Model (Shiu et al., Nature 2024), identical to pipeline/lif.py:

      dv/dt = (v_0 - v + g) / t_mbr      unless refractory
      dg/dt = -g / tau                   unless refractory
      spike when v > v_th  ->  v = v_rst, g = 0, refractory t_rfc
      presynaptic spike, after t_dly:  g = g + w     (w from FlyWire, in mV)

  Integration is exponential-Euler, which is exact for this linear system, so
  the only departure from Brian2 is that spike times land on the dt grid and
  the 1.8 ms delay is rounded to a whole number of steps.

  Cost per step is proportional to the number of *perturbed* neurons, not to
  the 2000-odd in the network: a neuron sitting at rest with nothing scheduled
  leaves the active list and is re-added the instant a synapse targets it.

  Step order matches pipeline/lif.py exactly:
      decay -> deliver delayed input -> Poisson drive -> threshold -> scatter
]]

local Brain = {}
Brain.__index = Brain

-- Constants in mV / ms (Kakaria & de Bivort 2017; Lazar 2021; Paul 2015)
local V_0    = -52.0
local V_RST  = -52.0
local V_TH   = -45.0
local T_MBR  = 20.0
local TAU    = 5.0
local T_RFC  = 2.2
local T_DLY  = 1.8
local W_POI  = 68.75   -- w_syn * f_poi, injected straight into v

-- Below this distance from rest a neuron counts as settled.  0.001 mV is
-- 1/7000 of the distance from rest to threshold.
local QUIET_EPS = 1e-3

local floor, exp = math.floor, math.exp

---------------------------------------------------------------------------
-- construction
---------------------------------------------------------------------------

-- net = { n, indptr, indices, weights, vrest }  -- flat 1-based arrays.
-- CSR by presynaptic neuron; indices are 1-based postsynaptic neuron ids.
-- vrest is the per-neuron resting potential: V_0 plus a constant offset
-- standing in for the synaptic input this truncated network lost, measured
-- from the whole-brain model (pipeline/compute_bias.py).
function Brain.new(net, dt)
  local self = setmetatable({}, Brain)
  dt = dt or 1.0
  self.dt = dt
  self.n = net.n
  self.indptr = net.indptr
  self.indices = net.indices
  self.weights = net.weights

  self.vrest = net.vrest
  if not self.vrest then
    self.vrest = {}
    for i = 1, net.n do self.vrest[i] = V_0 end
  end

  self.ev = exp(-dt / T_MBR)
  self.eg = exp(-dt / TAU)
  local a = 1.0 / T_MBR - 1.0 / TAU
  self.cvg = (self.eg - self.ev) / (T_MBR * a)
  self.rfcSteps = floor(T_RFC / dt + 0.5)
  self.delay = floor(T_DLY / dt + 0.5)
  if self.delay < 1 then self.delay = 1 end
  self.ring = self.delay + 1

  local n = self.n
  self.v, self.g, self.rfc = {}, {}, {}
  self.inAct, self.pendCount = {}, {}
  self.hz, self.pev = {}, {}
  self.spikeCount, self.rate = {}, {}
  local vrest = self.vrest
  for i = 1, n do
    self.v[i] = vrest[i]; self.g[i] = 0.0; self.rfc[i] = 0
    self.inAct[i] = false; self.pendCount[i] = 0
    self.hz[i] = 0.0; self.pev[i] = 0.0
    self.spikeCount[i] = 0; self.rate[i] = 0.0
  end

  -- double-buffered active list
  self.act, self.act2 = {}, {}
  self.nact = 0

  -- delay ring: value buffer plus an occupancy flag, so a neuron that is
  -- waiting for a delivery is never dropped from the active list
  self.pending, self.pendSet = {}, {}
  for d = 1, self.ring do
    local buf, flag = {}, {}
    for i = 1, n do buf[i] = 0.0; flag[i] = false end
    self.pending[d] = buf
    self.pendSet[d] = flag
  end
  self.slot = 1

  self.driven, self.ndriven = {}, 0   -- indices with hz > 0, for cheap clearing
  self.rngState = 0x2545F491
  self.steps = 0
  return self
end

function Brain:reset()
  local n = self.n
  local vrest = self.vrest
  for i = 1, n do
    self.v[i] = vrest[i]; self.g[i] = 0.0; self.rfc[i] = 0
    self.inAct[i] = false; self.pendCount[i] = 0
    self.hz[i] = 0.0; self.pev[i] = 0.0
    self.spikeCount[i] = 0; self.rate[i] = 0.0
  end
  for d = 1, self.ring do
    local buf, flag = self.pending[d], self.pendSet[d]
    for i = 1, n do buf[i] = 0.0; flag[i] = false end
  end
  self.nact = 0
  self.ndriven = 0
  self.slot = 1
  self.steps = 0
end

---------------------------------------------------------------------------
-- xorshift32, so the simulation never touches the game's RNG stream
---------------------------------------------------------------------------
function Brain:seed(s)
  s = s & 0xFFFFFFFF
  self.rngState = (s ~= 0) and s or 0x2545F491
end

---------------------------------------------------------------------------
-- input
---------------------------------------------------------------------------

function Brain:clearInput()
  local hz, pev, driven = self.hz, self.pev, self.driven
  for k = 1, self.ndriven do
    local i = driven[k]
    hz[i] = 0.0
    pev[i] = 0.0
  end
  self.ndriven = 0
end

-- Drive neuron `i` with a Poisson process at `rateHz`.  As in the paper each
-- event dumps W_POI into v -- ten times the distance to threshold -- so a
-- driven neuron fires at essentially the requested rate.
--
-- We store P(at least one event this step) = 1 - exp(-rate*dt) rather than
-- sampling a Poisson count.  Because one event already drives v far past
-- threshold, the spike outcome is identical to the reference implementation's
-- Poisson draw, so this is exact for the model rather than an approximation.
function Brain:setInput(i, rateHz)
  if rateHz <= 0.0 then return end
  local hz = self.hz
  if hz[i] == 0.0 then
    local k = self.ndriven + 1
    self.ndriven = k
    self.driven[k] = i
  end
  hz[i] = hz[i] + rateHz
  self.pev[i] = 1.0 - exp(-hz[i] * self.dt / 1000.0)
  if not self.inAct[i] then
    self.inAct[i] = true
    local k = self.nact + 1
    self.nact = k
    self.act[k] = i
  end
end

---------------------------------------------------------------------------
-- the step
---------------------------------------------------------------------------

function Brain:step()
  local nact = self.nact
  if nact == 0 then
    self.slot = self.slot % self.ring + 1
    self.steps = self.steps + 1
    return
  end

  local cur, nxt = self.act, self.act2
  local v, g, rfc, vrest = self.v, self.g, self.rfc, self.vrest
  local inAct, pendCount, pevArr = self.inAct, self.pendCount, self.pev
  local indptr, indices, weights = self.indptr, self.indices, self.weights
  local spikeCount = self.spikeCount
  local EV, EG, CVG = self.ev, self.eg, self.cvg
  local RFCS = self.rfcSteps
  local ring = self.ring

  local slot = self.slot
  local pend, pset = self.pending[slot], self.pendSet[slot]
  local dslot = slot + self.delay
  if dslot > ring then dslot = dslot - ring end
  local target, tset = self.pending[dslot], self.pendSet[dslot]

  local rng = self.rngState
  local nn = 0

  for k = 1, nact do
    local i = cur[k]
    local r = rfc[i]
    local vi, gi = v[i], g[i]
    local driveP = pevArr[i]

    -- the paper zeroes the refractory period of Poisson targets
    if driveP > 0.0 then r = 0; rfc[i] = 0 end

    if r > 0 then
      rfc[i] = r - 1
    else
      local vr = vrest[i]
      vi = vr + (vi - vr) * EV + gi * CVG
      gi = gi * EG
    end

    if pset[i] then
      gi = gi + pend[i]
      pend[i] = 0.0
      pset[i] = false
      pendCount[i] = pendCount[i] - 1
    end

    if driveP > 0.0 then
      rng = rng ~ ((rng << 13) & 0xFFFFFFFF)
      rng = rng ~ (rng >> 17)
      rng = rng ~ ((rng << 5) & 0xFFFFFFFF)
      rng = rng & 0xFFFFFFFF
      if rng * (1.0 / 4294967296.0) < driveP then
        vi = vi + W_POI
      end
    end

    if vi > V_TH and r <= 0 then
      vi = V_RST
      gi = 0.0
      rfc[i] = RFCS
      spikeCount[i] = spikeCount[i] + 1
      local hi = indptr[i + 1] - 1
      for e = indptr[i], hi do
        local j = indices[e]
        target[j] = target[j] + weights[e]
        if not tset[j] then
          tset[j] = true
          pendCount[j] = pendCount[j] + 1
        end
        if not inAct[j] then
          inAct[j] = true
          nn = nn + 1
          nxt[nn] = j
        end
      end
    end

    v[i] = vi; g[i] = gi

    local dv = vi - vrest[i]; if dv < 0 then dv = -dv end
    local ag = gi;       if ag < 0 then ag = -ag end
    if dv > QUIET_EPS or ag > QUIET_EPS or rfc[i] > 0
       or pendCount[i] > 0 or driveP > 0.0 then
      nn = nn + 1
      nxt[nn] = i
    else
      inAct[i] = false
    end
  end

  self.rngState = rng
  self.act, self.act2 = nxt, cur
  self.nact = nn
  self.slot = slot % ring + 1
  self.steps = self.steps + 1
end

function Brain:run(steps)
  for _ = 1, steps do self:step() end
end

---------------------------------------------------------------------------
-- read-out
---------------------------------------------------------------------------

-- Consume the spikes accumulated since the last call and fold them into a
-- 50 ms exponential rate estimate.  Returns the population mean in Hz.
function Brain:popRate(ids, stepsElapsed)
  local count = #ids
  if count == 0 or stepsElapsed <= 0 then return 0.0 end
  local rate, spikeCount = self.rate, self.spikeCount
  local window = self.dt * stepsElapsed
  local decay = exp(-window / 50.0)
  local gain = (1.0 - decay) * (1000.0 / window)
  local sum = 0.0
  for k = 1, count do
    local i = ids[k]
    local r = rate[i] * decay + spikeCount[i] * gain
    spikeCount[i] = 0
    rate[i] = r
    sum = sum + r
  end
  return sum / count
end

function Brain:activeCount() return self.nact end

return Brain
