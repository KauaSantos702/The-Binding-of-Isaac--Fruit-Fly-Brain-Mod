# Connectome — an Isaac familiar flown by a real fly's brain

A *Binding of Isaac: Repentance+* mod. It adds one passive item, **Connectome**,
which grants a familiar fly.

The fly has no state machine, no steering behaviours and no pathfinding. Every
frame the mod turns the room into a visual scene, injects it as Poisson spike
trains into **2053 real *Drosophila melanogaster* neurons wired by 83 791 real
synapses** taken from the FlyWire whole-brain connectome, integrates 33 ms of
leaky integrate-and-fire dynamics, and reads the fly's next move out of its
descending neurons.

Where the fly goes is whatever DNa02, DNp09, DNp01 and the rest happen to be
doing.

```
  big or fast-expanding ──▶ LC4 / LPLC2 / LC6 / LPLC1  (looming detectors)
    monsters, tears, fire, walls, a moving Isaac
  small and slow        ──▶ LC11 / LC17 / LC18         (small object motion)
    distant fly enemies, pickups, rocks  -- no escape from these
  poop, a still Isaac   ──▶ LC9 / LC10a / LC10d        (object tracking)
  the fly's own motion  ──▶ HS / VS / H2               (optic flow)
                                    │
                          PVLP / PLP / LAL premotor
                                    │
                    DNp09  DNp01  DNa01-03  DNb01-02  …
                                    │
                          turn, forward speed, escape
                                    │
                        ▲───────────┴── the fly moves, which
                        └────────────── regenerates its own optic flow
```

## Install

Copy `mod/` into the Isaac mods folder as `fly-brain-connectome`, or run:

```sh
sh tools/install.sh
```

Mods folder on this machine:
`C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\mods\`

Enable it in the in-game mods menu. Requires Repentance or Repentance+; no
other mods and no REPENTOGON.

## Using it

Pick up **Connectome** (Treasure Room, Boss Rush and Secret Room pools) and a
fly appears. It damages enemies on contact, harder while it is escaping.

With the debug console (`--luadebug` or the console-enabling mod):

| command | what it does |
|---|---|
| `flybrain` | cycle the overlay: compact -> full -> off |
| `flybrain compass` | switch on the MeTu → TuBu → ER → EPG landmark pathway (off by default; see below) |
| `flybrain reset` | return every brain to rest |

## Is it really a fly's brain?

Yes, with four caveats stated plainly.

**What is real.** Every neuron is a real FlyWire v783 cell with its real cell
type, and every connection is a real one with its real signed synapse count,
scaled by the free parameter `w_syn = 0.275 mV` from Shiu et al. (*Nature* 2024)
— the model that reproduces 91% of measured *Drosophila* sensorimotor responses.
Nothing about the wiring is invented, hand-tuned or learned. The Lua simulator
reproduces the numpy reference implementation at **rate correlation 0.994**.

**Caveat 1 — it is 1.5% of a brain.** 2053 of 138 639 neurons. Each kept neuron
loses about 60% of its real input, and what it loses is net inhibitory, so the
premotor and descending layers fire about 3× harder than they do in the full
model. The same neurons respond, in the same left/right pattern; the absolute
rates run hot. Adding neurons back was tried and measurably made it worse
(`docs/METHODS.md` §5).

**Caveat 2 — the connectome stops at the neck.** FlyWire has the brain, not the
ventral nerve cord, so how a descending neuron's spikes become leg movement is
not in the data. Three sign conventions are asserted from published fly
behaviour and are listed in `mod/fly_brain/motor.lua`.

**Caveat 3 — she has no sense of smell.** Poop would be the obvious thing for a
fly to smell, and the connectome has the whole olfactory pathway with known
chemical identities. It was tested in the full model before being built, and it
does not work: vinegar, geosmin and heat all produce the *same* descending
output (cosine similarity 0.99 at every drive rate from 10 to 150 Hz), and 73%
of Kenyon cells fire where a real fly fires 5-10%. The mod therefore has no
olfaction and no thermosensation — she finds poop by looking at it.

**Caveat 4 — the compass does not work.** The circuit contains the fly's whole
heading system, and the landmark pathway measurably carries the landmark's
direction as far as the ring neurons. But the EPG ring attractor never forms a
bump: `ER → EPG` is purely inhibitory (−9979 mV over 6657 connections), so a
landmark can only carve a hole in excitation that this model never supplies.
The Shiu model starts from total silence by design, and a ring attractor cannot
bootstrap from silence. So the fly navigates entirely on its looming and
object-motion pathways. The heading system ships anyway and is off by default.

## What it does, measured

`tools/behaviour.lua` runs the mod's own agent code headless. A threat closed in
from ten bearings:

| bearing | net turn | peak escape DN |
|---|---|---|
| −100° | **+226°** | 39.1 Hz |
| −40° | **+165°** | 51.8 Hz |
| +40° | **−190°** | 47.4 Hz |
| +100° | **−192°** | 39.3 Hz |

The fly turns away from the threat wherever it comes from — mean mirrored
asymmetry **+349°** — and the escape neurons peak for frontal looming, which is
LC4's and LPLC2's real tuning. None of that is coded anywhere; it is what the
connectome does.

She also turns *toward* things, which is a separate computation and was
measured separately: a target on the LC9/LC10 tracking channel pulls her
heading after it at all six bearings tested, while the small-object channel
does not. So standing still lets her close on you (150 → 24 px in 70 frames,
never fleeing), and charging at her sets the escape neurons to 44.6 Hz and she
bolts.

Cost: **8.3% of an Isaac tick idle, 17.7% with threats on screen.** Only
perturbed neurons are simulated, so a resting network is nearly free.

## Layout

```
mod/                  the mod, as installed
  main.lua            room -> percepts, movement command -> Velocity. Isaac API only here
  fly_brain/
    brain.lua         the LIF simulator (no Isaac API)
    agent.lua         sense -> think -> act, testable headless
    sensory.lua       scene -> spikes, by retinotopic azimuth
    motor.lua         descending spikes -> movement, with the VNC assumptions
    loader.lua        base64 + string.unpack decoder
    net_csr.lua       manifest for the connectome, split over 21 small part files
    net_ind*/net_weights*  2053 neurons, 83 791 synapses, base64 (455 KB)
    hud.lua           bars and graphs, drawn from one tinted pixel sprite
    net_meta.lua      sensory map and motor read-out groups
    net_vrest.lua     per-neuron resting potentials

pipeline/             extraction and validation (Python)
  circuit_spec.py     which cell types, and why, with citations
  build_circuit.py    FlyWire v783 -> build/circuit.json
  lif.py              reference implementation of the Shiu model
  validate.py         Lua vs numpy, pruned vs full, sub vs whole brain
  probe_compass.py    why the heading system produces no behaviour
  probe_new_senses.py / probe_olfaction_gain.py   why smell and heat do not work
  export_lua.py       pack for the mod

tools/
  bench.lua           cost, and the assertion that the network is silent at rest
  behaviour.lua       headless fly assays
  install.sh
```

`docs/METHODS.md` has the full record: what was extracted, what was assumed,
every validation number, and the three corrections that were tried (one worked
a little, one did nothing, one made things worse).

## Credits

- **Connectome and whole-brain model** — [`eonsystemspbc/fly-brain`](https://github.com/eonsystemspbc/fly-brain),
  implementing Shiu et al., *A leaky integrate-and-fire model based on the
  connectome of the entire adult Drosophila brain*, Nature 2024.
- **FlyWire v783** — Dorkenwald et al., 2024.
- **Cell-type annotations** — Schlegel et al., 2024,
  [`flyconnectome/flywire_annotations`](https://github.com/flyconnectome/flywire_annotations).
