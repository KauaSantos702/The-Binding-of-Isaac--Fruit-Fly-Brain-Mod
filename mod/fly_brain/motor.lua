--[[ motor.lua -- turn descending-neuron spikes into movement.

  The connectome ends at the neck: FlyWire contains the brain, not the ventral
  nerve cord, so how a descending neuron's spikes become leg movement is *not*
  in the data.  The sign conventions below come from the behavioural
  literature and are the one place in this mod where something outside the
  connectome is asserted.  They are listed explicitly so they can be argued
  with:

    DNa01, DNa02, DNa03   ipsilateral turning.  Right DNa02 turns the fly
                          right.  (Rayshubskiy et al. 2020; Yang et al. 2023)

    DNp09                 stops forward walking and can drive backward
                          walking; freeze/escape.  (Cande et al. 2018)

    DNp01 (giant fibre)   short-latency escape takeoff.
                          (von Reyn et al. 2014)

    DNp04, DNp06, DNp10,  looming-responsive escape descending neurons; the
    DNp11, DNp35, DNp103  fly turns *away* from the side that fires.
                          (Namiki et al. 2018)

    DNb01, DNb02, DNa04,  forward walking drive.
    DNg13

  Everything upstream of these neurons -- which of them fire, how strongly, in
  what left/right balance -- is computed by the connectome simulation and is
  not scripted anywhere.
]]

local Motor = {}
Motor.__index = Motor

function Motor.new(meta)
  local self = setmetatable({}, Motor)
  self.m = meta.motor
  self.out = {
    turn = 0.0, forward = 0.0, escape = 0.0,
    steerL = 0.0, steerR = 0.0, escL = 0.0, escR = 0.0,
    dnL = 0.0, dnR = 0.0,
  }
  return self
end

-- Read the spikes accumulated over `steps` and produce movement commands.
function Motor:read(brain, steps)
  local m, o = self.m, self.out
  o.steerL = brain:popRate(m.steer_left, steps)
  o.steerR = brain:popRate(m.steer_right, steps)
  o.escL   = brain:popRate(m.escape_left, steps)
  o.escR   = brain:popRate(m.escape_right, steps)
  o.dnL    = brain:popRate(m.dn_left, steps)
  o.dnR    = brain:popRate(m.dn_right, steps)
  o.fwdRate = brain:popRate(m.forward, steps)

  -- Goal-directed steering: ipsilateral, so right minus left.
  local goal = (o.steerR - o.steerL)

  -- Escape steering: the fly turns away from the eye that is screaming, so the
  -- sign is flipped relative to the goal channel.
  local flee = (o.escL - o.escR)

  -- The three named steering DNs per side are a thin read-out, so the whole
  -- descending population's left/right imbalance is folded in as well.  It
  -- carries the same information -- the probe experiments show it swinging
  -- from 8 Hz to 40 Hz between eyes under a one-sided looming stimulus.
  local bulk = (o.dnL - o.dnR)

  o.escape = 0.5 * (o.escL + o.escR)
  o.turn = 0.05 * goal + 0.022 * flee + 0.012 * bulk
  o.forward = o.fwdRate
  return o
end

return Motor
