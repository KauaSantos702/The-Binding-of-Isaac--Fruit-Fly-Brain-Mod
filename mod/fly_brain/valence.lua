--[[ valence.lua -- what she has learned about the world.

  One bar, running from terrified to bold, moved by consequences.  Hurting
  something and getting away with it pushes her towards courage; being burned
  pushes her back towards fear.  Where the bar sits decides how she reads the
  room: the same Gaper is either something to run from or something to chase.

  WHERE THIS SITS, AND WHAT IT IS NOT
  -----------------------------------
  The connectome does not learn.  The Shiu model has fixed weights and no
  plasticity at all, so nothing in this file is the fly's own learning, and it
  is not claimed to be.

  But the *place* it acts is real.  A fly learns valence in the mushroom body:
  dopaminergic neurons signalling punishment or reward reweight the Kenyon cell
  synapses onto MBONs, and the MBONs bias approach against avoidance -- without
  changing what any sensory neuron detects.  Nothing about a looming disc looks
  different to a trained fly; what changes is which way the decision falls.

  That is exactly what this does.  It never touches the sensory encoding or the
  network.  It moves a handful of gains that tilt the balance between two
  channels the connectome already computes, plus how hard she flies:

      LC4 / LPLC2  ->  DNp09        the escape channel
      LC9 / LC10a                   the pursuit channel

  (The mushroom body itself is in the extracted circuit's data but unusable --
  pipeline/probe_new_senses.py measured 73% of Kenyon cells firing where a real
  fly fires 5-10%.  Even a working one would not learn here, for want of
  plasticity.)

  There is also a second, smaller memory: what specifically hurt her.  Fire
  that has burned her once reads as more dangerous afterwards, while enemies
  she has beaten read as less.  That keeps "teaching her" from collapsing into
  one global mood.

  WATCHING SOMEONE ELSE WIN
  -------------------------
  Courage earned only by landing hits is a trap: frightened, she will not go
  near anything, so she cannot land a hit, so she stays frightened.  The only
  way out was waiting for the mood to fade, and one more burn undid it.

  So a kill she had no part in still counts for something, as long as it
  happened where she could see it.  Two things make that reasonable.
  Observational learning is real in Drosophila -- flies take on what they have
  watched other flies do, from mate choice to avoiding a danger they never met
  themselves.  And more simply, this is safety learning: the thing that was
  frightening her stopped existing, and nothing bad happened to her.
]]

local Valence = {}

-- How far one event moves the bar.  Killing is worth much more than a scratch,
-- and being burned is worth more than either -- a fly that learns caution
-- slowly and boldness quickly does not last long.
Valence.REWARD = {
  hit = 0.05,          -- landed a contact hit and walked away
  kill = 0.18,         -- something she had hurt died
  -- Enough that watching Isaac clear a room genuinely resets a scare: a
  -- burned fly is at -0.80, and a roomful of kills has to be able to undo it.
  -- Still well under what doing it herself is worth.
  witness = 0.10,      -- something died nearby and it was not her doing
}
Valence.PUNISH = {
  burn = 0.40,         -- fire
  shot = 0.18,         -- hit by a projectile
}

-- Half-lives, in frames at 30 fps.  The mood fades in about fifteen seconds so
-- she drifts back to her own baseline; what she learned about a *specific*
-- thing lasts four times longer.
local MOOD_HALFLIFE = 450
local MEMORY_HALFLIFE = 1800

local MOOD_DECAY = 0.5 ^ (1 / MOOD_HALFLIFE)
local MEMORY_DECAY = 0.5 ^ (1 / MEMORY_HALFLIFE)

local function clamp(v, lo, hi)
  if v < lo then return lo elseif v > hi then return hi end
  return v
end

-- Linear between three points, so the neutral value is set independently of
-- the two extremes rather than falling halfway between them.
local function tilt(c, atFear, atNeutral, atCourage)
  if c >= 0 then
    return atNeutral + (atCourage - atNeutral) * c
  end
  return atNeutral + (atNeutral - atFear) * c
end

---------------------------------------------------------------------------

function Valence.new()
  return {
    courage = 0.0,       -- -1 terrified .. +1 bold
    memory = {},         -- what she has learned about particular things
    lastEvent = nil,     -- for the overlay
    lastEventAge = 999,
  }
end

function Valence.reward(state, kind, key)
  local amount = Valence.REWARD[kind] or 0.0
  state.courage = clamp(state.courage + amount, -1.0, 1.0)
  if key then
    state.memory[key] = clamp((state.memory[key] or 0.0) + amount * 1.5, -1.0, 1.0)
  end
  state.lastEvent = kind
  state.lastEventAge = 0
  return state.courage
end

function Valence.punish(state, kind, key)
  local amount = Valence.PUNISH[kind] or 0.0
  state.courage = clamp(state.courage - amount, -1.0, 1.0)
  if key then
    state.memory[key] = clamp((state.memory[key] or 0.0) - amount * 1.5, -1.0, 1.0)
  end
  state.lastEvent = kind
  state.lastEventAge = 0
  return state.courage
end

-- Call once a frame.
function Valence.decay(state)
  state.courage = state.courage * MOOD_DECAY
  if state.courage < 0.0005 and state.courage > -0.0005 then
    state.courage = 0.0
  end
  for k, v in pairs(state.memory) do
    local n = v * MEMORY_DECAY
    state.memory[k] = (n < 0.002 and n > -0.002) and nil or n
  end
  state.lastEventAge = state.lastEventAge + 1
end

---------------------------------------------------------------------------
-- how the bar reads the room
---------------------------------------------------------------------------

-- The gains the agent takes.  None of these change what the eye sees -- they
-- change which of the two channels wins, and how hard she is willing to fly.
function Valence.gains(state)
  local c = state.courage
  return {
    -- how much an enemy is worth to the pursuit cells
    huntWeight = tilt(c, 0.05, 0.45, 5.00),
    -- and how much of a threat that same enemy still looks like
    preyThreat = tilt(c, 1.50, 1.00, 0.12),
    -- how hard the escape neurons have to shout before she runs
    escapeThreshold = tilt(c, 14.0, 22.0, 42.0),

    -- Committed, she flies harder to get there.  Speed on its own makes her
    -- *worse* -- measured: the turning radius is speed over turn rate, so a
    -- faster fly on the same steering swings wide and sails past what she is
    -- chasing, and the hits drop by a quarter.  She has to bank harder too,
    -- which is what an animal in fast pursuit does.  The fear end of these is
    -- deliberately the same as neutral: a frightened fly still needs to be
    -- able to bolt.
    baseSpeed = tilt(c, 2.6, 2.6, 4.2),
    maxSpeed  = tilt(c, 9.0, 9.0, 15.0),
    maxTurn   = tilt(c, 0.30, 0.30, 0.50),
    turnGain  = tilt(c, 0.085, 0.085, 0.130),
  }
end

-- What she has learned about one particular kind of thing, as a multiplier on
-- how threatening it looks.  Above 1 means it has hurt her before.
function Valence.threatScale(state, key)
  local m = key and state.memory[key]
  if not m then return 1.0 end
  if m < 0 then return 1.0 - m * 0.9 end      -- burned once: up to 1.9x
  return 1.0 - m * 0.55                        -- beaten often: down to 0.45x
end

function Valence.mood(state)
  local c = state.courage
  if c > 0.55 then return "bold"
  elseif c > 0.15 then return "game"
  elseif c > -0.15 then return "wary"
  elseif c > -0.55 then return "nervous"
  else return "terrified" end
end

function Valence.serialise(state)
  return { courage = state.courage, memory = state.memory }
end

function Valence.restore(state, saved)
  if type(saved) ~= "table" then return state end
  state.courage = clamp(tonumber(saved.courage) or 0.0, -1.0, 1.0)
  if type(saved.memory) == "table" then
    state.memory = {}
    for k, v in pairs(saved.memory) do
      local n = tonumber(v)
      if n then state.memory[k] = clamp(n, -1.0, 1.0) end
    end
  end
  return state
end

return Valence
