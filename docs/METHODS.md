# How the fly got its brain

This is the record of what was extracted from the FlyWire connectome, what was
assumed, what was measured, and what did not work. Every number below is
reproducible from the scripts in `pipeline/` and `tools/`.

---

## 1. Sources

| What | Where | Size |
|---|---|---|
| Whole-brain LIF model + connectivity | [`eonsystemspbc/fly-brain`](https://github.com/eonsystemspbc/fly-brain) (FlyWire v783) | 138 639 neurons, 15 091 983 signed edges |
| Cell-type annotations | [`flyconnectome/flywire_annotations`](https://github.com/flyconnectome/flywire_annotations) (Schlegel et al. 2024) | 138 625 of 138 639 matched |
| Neuron model | Shiu et al., *Nature* 2024 | leaky integrate-and-fire, 91% agreement with fly experiments |

The `fly-brain` repository ships the graph and the signs (excitatory /
inhibitory per connection) but no cell-type names, so the annotation table was
downloaded separately and joined on FlyWire root id.

**No connection weight in this project is invented.** Every weight is
`Excitatory x Connectivity × 0.275 mV` straight out of the v783 parquet. The
only operations applied are selection, population subsampling with weight
rescaling, and one measured per-neuron offset (§5).

---

## 2. The model

Exactly the model in `fly-brain/code/paper-phil-drosophila/model.py`, restated
in `pipeline/lif.py` and transcribed into `mod/fly_brain/brain.lua`:

```
dv/dt = (v_0 - v + g) / t_mbr        unless refractory
dg/dt = -g / tau                     unless refractory
spike when v > v_th  ->  v = v_rst, g = 0, refractory t_rfc
presynaptic spike, after t_dly:  g = g + w
```

with `v_0 = v_rst = -52 mV`, `v_th = -45 mV`, `t_mbr = 20 ms`, `tau = 5 ms`,
`t_rfc = 2.2 ms`, `t_dly = 1.8 ms`, `w_syn = 0.275 mV` per synapse. Integration
is exponential-Euler, which is exact for this linear system; the only departure
from Brian2 is that spike times land on the 1 ms grid and the 1.8 ms delay
rounds to 2 steps.

Sensory input is the paper's own optogenetic-activation mechanism: a Poisson
process that injects `w_syn × f_poi = 68.75 mV` directly into `v`. That is ten
times the distance from rest to threshold, so a driven neuron fires at
essentially the requested rate.

---

## 3. Which neurons, and why

`pipeline/circuit_spec.py` selects one complete sensorimotor loop. Populations
larger than their cap are subsampled evenly along the anterior-posterior axis
so retinotopic coverage is preserved, and the survivors' outgoing weights are
scaled by `n_total / n_kept` so downstream drive is unchanged in expectation.

| Role | Cell types | Neurons |
|---|---|---|
| Looming detectors | LC4, LPLC2, LC6, LPLC1, LC16, LPLC4 | 152 |
| Small-object motion | LC11, LC17, LC18, LC12, LC13, LC15, LC21, LC22 | 128 |
| Object position | LC9, LC10a, LC10d | 52 |
| Wide-field optic flow | HSN, HSE, HSS, H2, VS1–6 | 20 |
| Landmark pathway | MeTu1–4, TuBu, TuTu | 284 |
| Heading system | EPG, EPGt, PEN_a, PEN_b, PEG, Delta7, EL, PFL1–3 | 172 + 50 |
| Rest of the central complex | ER1–6, PFN\*, hDelta\*, vDelta\*, FC\*, FR\*, LNO\*, ExR\* | 776 |
| Descending neurons | DNa01–04, DNp01, DNp09, DNp04/06/10/11, DNb\*, DNg13 + 53 selected by connectivity | 79 |
| Bridge interneurons | selected by connectivity: ≥90 synapses both into and out of the core, two rounds | 340 |

**Total: 2053 neurons, 116 078 connections.** Pruning connections below
0.55 mV (two synapse-equivalents) leaves **83 791**, which is what ships.

### Retinotopy

FlyWire coordinates were identified empirically from landmarks in the
annotation table (`pos_x` left→right: left-side neurons mean 81 084 vs right
180 381; `pos_y` dorsal→ventral: central complex 40 924 vs SEZ motor neurons
79 893; `pos_z` anterior→posterior: antennal ORNs 820 vs mushroom body 4 626).
Within one optic lobe the lobula is a retinotopic sheet, so a neuron's `z`
maps to visual azimuth and its `y` to elevation. Each eye is assigned 170° of
coverage, from 10° across the midline to 160° behind, giving the binocular
overlap in front that a real fly has.

This is the one piece of *anatomical* interpretation in the pipeline. It is a
modelling assumption, not a measurement.

---

## 4. Validation

### A. The Lua engine reproduces the numpy reference

Same network, same six stimulus protocols, 8 trials each, independent RNG
streams — so per-neuron firing rates are compared, the measure
`fly-brain/code/compare_ground_truth.py` itself uses.

| Protocol | Jaccard | rate correlation | total-rate ratio |
|---|---|---|---|
| loom_right | 0.997 | 0.9917 | 0.992 |
| loom_left | 0.997 | 0.9944 | 0.983 |
| loom_front | 0.982 | 0.9907 | 0.985 |
| landmark_front | 0.981 | 0.9983 | 1.003 |
| landmark_right | 0.991 | 0.9981 | 0.994 |
| flow_yaw | 1.000 | 0.9906 | 0.997 |
| **mean** | **0.991** | **0.9940** | |

Reproduce with `python pipeline/validate.py --only a`.

### B. Pruning single-synapse connections is free

116 078 → 83 791 connections keeps 97.3% of the synaptic mass and changes
nothing measurable: rate correlation **0.9999** on every protocol
(`validate.py --only b`).

### C. The circuit versus the whole 138 639-neuron brain

The full model was run on the same stimuli and compared on the 2053 neurons
the circuit contains (`validate.py --only c`, `score_vs_full.py`):

- **Poisson-driven sensory neurons: ratio 1.02, r = 0.99.** Exact, as expected.
- **Central complex: ratio 0.96.** Nearly exact — it keeps 93% of its real
  input, so truncation barely touches it.
- **Downstream premotor and descending neurons: same neurons fire (Jaccard
  0.67–0.78, r = 0.74–0.88) but about 3× too hard.**

The over-activity is truncation: a kept neuron loses on average 60% of its
real synaptic input, and what it loses is net inhibitory.
`pipeline/diagnose_gain.py` localises it precisely — the excess is entirely in
neurons with low input coverage (bridge 2.9×, descending 2.1×), and zero in the
central complex.

Because the motor decoder reads left/right *differences* and thresholds, a
uniform gain on the downstream layer is largely absorbed. It is still a real
discrepancy and it is why the mod tunes its turn gains rather than using the
firing rates in absolute units.

---

## 5. Two corrections, and one that failed

**A global synaptic gain does not work.** Sweeping `w_syn × g` for
`g ∈ [0.70, 1.00]` moves the rate ratio only from 1.37 to 1.25, because the
excess is not recurrent amplification (`pipeline/calibrate_gain.py`).

**Adding the missing neurons back makes it worse.** `find_recruits.py` runs the
whole brain and lists the 900 neurons it activates that the circuit lacks —
mostly PVLP/PLP optic-glomerulus interneurons and 100 more descending neurons.
Adding them took the circuit to 2953 neurons and 139 618 connections and pushed
the downstream ratio from 2.6 to **6.0**, with Jaccard falling 0.74 → 0.67.
Every added shell arrives without *its own* inhibitory inputs, so it excites
the circuit without braking it. Kept in the pipeline behind
`--max-recruits` (default 0) as a documented negative result.

**Replacing the missing input with its measured mean effect helps a little.**
For a dropped presynaptic neuron *j* firing at rate ν*ⱼ* into neuron *i* with
weight *wⱼᵢ*, the mean synaptic drive contributed is `wⱼᵢ · νⱼ · tau`. Summing
over all dropped inputs gives a constant offset to each neuron's resting
potential. ν*ⱼ* is not assumed — it is the rate that neuron actually had in the
whole-brain simulation of the same stimuli. The single fitted parameter is a
scalar *k* on that offset.

The measurement independently confirms the diagnosis: mean offset is
**+0.006 mV for the central complex** (which already matched) and
**−1.345 mV for descending neurons** (which ran hottest).

Fitting *k* (`pipeline/fit_bias.py`) gives **k = 0.75**, improving downstream
correlation 0.607 → 0.630 and the ratio 3.30 → 3.21. Small, free at runtime,
and kept. It does not close the gap because the missing inhibition is phasic
and stimulus-locked, while a resting-potential offset is static.

Offsets are clamped so no resting potential comes within 2 mV of threshold;
the network must stay silent without input, which `tools/bench.lua` asserts.

---

## 6. The compass does not work, and here is why

The circuit contains the fly's entire heading system — 948 of its 2053
neurons. `pipeline/probe_compass.py` establishes three things:

1. **The landmark pathway is intact and azimuth-selective as far as the ring
   neurons.** Moving a landmark around the visual field measurably changes
   which ER neurons fire: activation-pattern similarity is 0.87 between
   adjacent azimuths and 0.71 between opposite ones. MeTu → TuBu → ER carries
   real spatial information.

2. **The EPG ring attractor never forms a bump.** With no input the population
   is silent. With uniform tonic drive all 47 EPG neurons switch on together
   at the drive rate, and the landmark changes the pattern by less than 2%.
   `ER → EPG` is **−9979 mV over 6657 connections** — purely inhibitory, as it
   must be, since ring neurons are GABAergic. A landmark can only carve a hole
   in pre-existing excitation, and this model supplies none.

3. **PFL3, the steering output, therefore carries no heading signal** (0.1–0.9 Hz,
   no dependence on landmark azimuth), and the mod does not use one.

This is a property of the Shiu et al. model rather than of the extraction. That
model deliberately starts from total silence in order to model optogenetic
activation from a defined baseline, and a ring attractor cannot bootstrap
itself out of silence — the bump needs ongoing background activity that the
model omits by design.

The pathway ships anyway, because it is really in the fly and it demonstrably
carries the signal as far as the ring neurons. It is **off by default** since
it costs roughly half the simulation's CPU for no behaviour; `flybrain compass`
in the console switches it on.

---

## 7. What the fly actually does, and where it comes from

`tools/behaviour.lua` runs the mod's own agent code in a headless arena.

**Turn response versus threat bearing** (net turning over 45 frames of a
closing threat held at a fixed bearing):

| bearing | net turn | peak escape DN |
|---|---|---|
| −140° | +191° | 38.9 Hz |
| −100° | +226° | 39.1 Hz |
| −70° | +205° | 41.4 Hz |
| −40° | +165° | 51.8 Hz |
| −15° | +91° | 49.8 Hz |
| +15° | −158° | 48.5 Hz |
| +40° | −190° | 47.4 Hz |
| +70° | −207° | 40.3 Hz |
| +100° | −192° | 39.3 Hz |
| +140° | −147° | 25.7 Hz |

The response is cleanly odd-symmetric — mean mirrored difference **+349°** —
so the fly turns away from a threat wherever it appears, and the escape
descending neurons peak for frontal looming, which is the tuning LC4 and LPLC2
actually have. None of this is scripted; it falls out of the connectome.

**In an enclosed room** with only walls and the player: 147 of 239 frames
contain a real turn, the fly covers the whole arena, and wall looming drives
escape on 71 frames.

### What is *not* from the connectome

FlyWire contains the brain, not the ventral nerve cord, so how a descending
neuron's spikes become leg movement is not in the data. Three sign conventions
are asserted from the behavioural literature, and they are listed in
`mod/fly_brain/motor.lua`:

- DNa01/02/03 drive **ipsilateral** turning (Rayshubskiy et al. 2020).
- Looming-responsive escape DNs turn the fly **away** from the firing side
  (Namiki et al. 2018).
- DNp09 stops forward walking; DNp01 (the giant fibre) drives takeoff.

Also not biology: the leash that keeps the familiar near the player, and the
contact-damage rule. Both are game constraints and are marked as such in the
source.

---

## 8. Cost

Measured with `tools/bench.lua` on Lua 5.3.6, 33 integration steps per Isaac
tick, so brain time runs at wall-clock speed:

| Scene | per frame | share of a 33.3 ms tick | active neurons |
|---|---|---|---|
| idle, player in view | 2.76 ms | 8.3% | 477 |
| player + 2 threats + a wall | 5.89 ms | 17.7% | 592 |
| same, landmark pathway on | 9.94 ms | 29.8% | 1200 |

Cost scales with the number of *perturbed* neurons, not the 2053 in the
network: a neuron at rest with nothing scheduled leaves the active list and
rejoins the moment a synapse targets it. The mod also throttles steps per
frame to stay inside a 6 ms budget, so with several flies the brain slows down
rather than the game.

Load time is 0.07 s for the whole 83 791-connection network.

---

## 9. Reproducing everything

```bash
python -m venv .venv && .venv/Scripts/python -m pip install pandas pyarrow numpy scipy Pillow

python pipeline/build_circuit.py     # extract from FlyWire v783
python pipeline/find_recruits.py     # whole-brain reference runs (slow, cached)
python pipeline/compute_bias.py      # measure the lost input
python pipeline/fit_bias.py          # fit k
python pipeline/export_lua.py        # pack into mod/fly_brain/
python pipeline/make_sprites.py      # sprites and .anm2

tools/lua53/lua53.exe tools/bench.lua        # cost + silence assertions
tools/lua53/lua53.exe tools/behaviour.lua    # behavioural assertions
python pipeline/validate.py                  # Lua vs numpy, pruned vs full
python pipeline/probe_compass.py             # the negative result in §6

sh tools/install.sh
```
