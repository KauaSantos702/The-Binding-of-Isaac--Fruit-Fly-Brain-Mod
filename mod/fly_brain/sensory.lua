--[[ sensory.lua -- turn the room into spikes.

  Every sensory neuron in the network carries a head-centred azimuth in
  degrees (negative = the fly's left) recovered from its retinotopic position
  in the optic lobe.  To present a stimulus we drive the neurons whose
  receptive fields point at it, with a Gaussian falloff -- the same encoding
  the offline validation protocols use, so what runs in the game is what was
  measured against the whole-brain model.

  Channels, and the real cell types behind them:

    loom      LC4, LPLC2, LC6, LPLC1, LC16, LPLC4
              Looming detectors.  In the fly these drive DNp09 and the giant
              fibre DNp01 and produce escape.  Fed here by enemies and
              projectiles, weighted by angular size and approach speed.

    object    LC11, LC17, LC18, LC12, LC13, LC15, LC21, LC22
              Small-object motion detectors.  Fed by pickups and by distant
              enemies.

    bar       LC9, LC10a, LC10d
              Object position / tracking.  Fed by the nearest enemy.

    landmark  MeTu1-4 -> TuBu -> ER -> EPG
              The sky-compass pathway a fly uses to anchor its heading to the
              world.  Fed by the player, who is the fly's landmark.

    flow      HSN, HSE, HSS, H2, VS1-6
              Wide-field optic flow.  Fed by the fly's *own* motion, which is
              what closes the loop: the fly's turning regenerates the flow that
              its own optomotor circuitry then corrects for.
]]

local Sensory = {}
Sensory.__index = Sensory

local exp, abs, floor = math.exp, math.abs, math.floor

-- Build per-channel lists sorted by azimuth so a stimulus only touches the
-- neurons whose receptive fields are near it.
function Sensory.new(meta)
  local self = setmetatable({}, Sensory)
  self.chan = {}
  for kind, list in pairs(meta.sensory) do
    local rows = {}
    for k = 1, #list do
      local e = list[k]
      rows[#rows + 1] = { i = e[1], az = e[2], el = e[3], side = e[4] }
    end
    table.sort(rows, function(a, b) return a.az < b.az end)
    local idx, az = {}, {}
    for k = 1, #rows do idx[k] = rows[k].i; az[k] = rows[k].az end
    self.chan[kind] = { idx = idx, az = az, n = #rows }
  end
  return self
end

-- first index whose azimuth >= target
local function lowerBound(az, n, target)
  local lo, hi = 1, n + 1
  while lo < hi do
    local mid = (lo + hi) // 2
    if az[mid] < target then lo = mid + 1 else hi = mid end
  end
  return lo
end

-- Drive the neurons of `kind` looking near `azDeg` with a Gaussian of width
-- `sigmaDeg` peaking at `peakHz`.
function Sensory:stimulate(brain, kind, azDeg, sigmaDeg, peakHz)
  local c = self.chan[kind]
  if not c or peakHz <= 0 then return 0 end
  local idx, az, n = c.idx, c.az, c.n
  local lo = lowerBound(az, n, azDeg - 3 * sigmaDeg)
  local inv = 1.0 / sigmaDeg
  local hiAz = azDeg + 3 * sigmaDeg
  local count = 0
  for k = lo, n do
    local a = az[k]
    if a > hiAz then break end
    local d = (a - azDeg) * inv
    local hz = peakHz * exp(-0.5 * d * d)
    if hz > 1.0 then
      brain:setInput(idx[k], hz)
      count = count + 1
    end
  end
  return count
end

-- Optic flow.  HS cells are excited by front-to-back motion across their own
-- eye.  Turning left sweeps the scene from front to back across the *right*
-- eye, so a leftward turn excites right HS, and vice versa; forward
-- translation excites both.  Driving them from the fly's own movement is what
-- gives the circuit a closed optomotor loop rather than an open-loop readout.
function Sensory:flow(brain, yawRate, forwardSpeed)
  local c = self.chan.flow
  if not c then return end
  local idx, az, n = c.idx, c.az, c.n
  local rightDrive = 55.0 * math.max(0.0, -yawRate) + 22.0 * forwardSpeed
  local leftDrive = 55.0 * math.max(0.0, yawRate) + 22.0 * forwardSpeed
  for k = 1, n do
    local hz = (az[k] >= 0) and rightDrive or leftDrive
    if hz > 1.0 then brain:setInput(idx[k], hz) end
  end
end

function Sensory:count(kind)
  local c = self.chan[kind]
  return c and c.n or 0
end

return Sensory
