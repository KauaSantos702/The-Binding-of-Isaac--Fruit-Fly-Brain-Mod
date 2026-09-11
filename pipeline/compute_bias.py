"""Replace each neuron's missing input with its measured mean effect.

The circuit keeps 2053 of 138 639 neurons, so a kept neuron loses on average
60% of its real synaptic input.  That loss is not balanced -- the dropped
neurons are net inhibitory onto exactly the populations that then run hot (see
diagnose_gain.py: downstream firing is 2.5x the full model's).

Adding the missing neurons back does not help, and measurably hurts: every
added shell arrives without *its* inhibitory inputs, so it excites the circuit
without braking it (score_vs_full.py with --max-recruits 900 gives a
downstream ratio of 6.0 instead of 2.5).

So instead of adding the neurons we add their average effect.  For a dropped
presynaptic neuron j firing at rate nu_j into neuron i with weight w_ji, the
mean of the synaptic variable g contributed is w_ji * nu_j * tau.  Summing over
all dropped inputs gives a constant offset to neuron i's resting potential:

    bias_i = k * tau * sum_{j dropped} w_ji * nu_j

Crucially nu_j is not assumed: it is the firing rate that neuron j actually had
in the whole-brain simulation of the same stimuli (cached by find_recruits.py).
The single free parameter is the scalar k, fitted by sweep.

The approximation is that bias_i is static, while the real missing input varies
with the stimulus.  We therefore average over all six protocols, and report in
docs/METHODS.md how much of the gap this closes.

Run:  python pipeline/compute_bias.py
Writes build/bias.npy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from lif import TAU
from netio import Circuit, ROOT


def main():
    c = Circuit()
    d = np.load(ROOT / "build/full_rates_all.npz")
    root_ids = d["root_ids"]
    N = len(root_ids)
    protos = [k for k in d.files if k != "root_ids"]

    pos = pd.Series(np.arange(N), index=root_ids)
    gidx = pos.reindex(c.root_id).to_numpy()
    kept = np.zeros(N, dtype=bool)
    kept[gidx] = True
    local_of = -np.ones(N, dtype=np.int64)
    local_of[gidx] = np.arange(c.N)

    print("loading the full connectome...")
    t = pq.read_table(ROOT / "fly-brain/data/2025_Connectivity_783.parquet",
                      columns=["Presynaptic_Index", "Postsynaptic_Index",
                               "Excitatory x Connectivity"])
    pre = t.column(0).to_numpy()
    post = t.column(1).to_numpy()
    w = t.column(2).to_numpy().astype(np.float64) * 0.275
    del t

    # edges from a dropped neuron into a kept one
    m = (~kept[pre]) & kept[post]
    pre_d, post_k, w_d = pre[m], local_of[post[m]], w[m]
    print(f"  {m.sum()} lost edges into the {c.N} kept neurons")

    tau_s = TAU / 1000.0
    per_proto = np.zeros((len(protos), c.N))
    for p, name in enumerate(protos):
        nu = d[name]
        contrib = w_d * nu[pre_d] * tau_s
        acc = np.zeros(c.N)
        np.add.at(acc, post_k, contrib)
        per_proto[p] = acc
        act = nu[pre_d] > 0.5
        print(f"  {name:16s} lost drive: mean {acc.mean():7.3f} mV, "
              f"min {acc.min():8.2f}, max {acc.max():7.2f}  "
              f"({np.unique(pre_d[act]).size} dropped neurons firing)")

    bias = per_proto.mean(axis=0)
    np.save(ROOT / "build/bias.npy", bias)
    print(f"\nmean bias {bias.mean():.3f} mV  (negative = the missing input was "
          f"net inhibitory)")
    print(f"  hyperpolarising for {(bias < 0).sum()} of {c.N} neurons, "
          f"depolarising for {(bias > 0).sum()}")
    for role in sorted(set(c.role)):
        msk = c.role == role
        print(f"    {role:16s} mean {bias[msk].mean():8.3f} mV")
    print("wrote build/bias.npy")


if __name__ == "__main__":
    main()
