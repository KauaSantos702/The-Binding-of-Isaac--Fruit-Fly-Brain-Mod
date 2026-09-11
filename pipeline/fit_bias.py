"""Fit the single scalar k on the measured missing-input bias.

    v_rest_i = V_0 + k * bias_i

k = 0 is the uncorrected truncated circuit; k = 1 applies the full measured
mean effect of the lost synaptic input.  Scored against the whole-brain model
on all six protocols.

Run:  python pipeline/fit_bias.py
Writes build/bias_fit.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

import protocols
from lif import LIFNetwork
from netio import Circuit, ROOT

TRIALS = 4
KS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def sub_rates(circuit, W, name, bias, trials=TRIALS):
    hz, no_rfc, ms = protocols.drive(circuit, name)
    acc = np.zeros(W.shape[0])
    for tr in range(trials):
        net = LIFNetwork(W, dt=1.0, seed=400 + tr, bias=bias)
        acc += net.run(ms, rate_fn=lambda _t: hz, no_refractory=no_rfc,
                       record=False)["counts"]
    return acc / trials / (ms / 1000.0)


def main():
    c = Circuit()
    W = sp.load_npz(ROOT / "build/W_pruned.npz")
    bias = np.load(ROOT / "build/bias.npy")
    d = np.load(ROOT / "build/full_rates_all.npz")
    root_ids = d["root_ids"]
    gidx = pd.Series(np.arange(len(root_ids)), index=root_ids).reindex(c.root_id).to_numpy()

    print(f"{c.N} neurons, {W.nnz} edges\n")
    print("   k   Jaccard   rate r   downstream ratio   downstream r")
    results = []
    for k in KS:
        b = bias * k
        js, rs, ratios, rdns = [], [], [], []
        for name in protocols.PROTOCOLS:
            full = d[name][gidx]
            got = sub_rates(c, W, name, b)
            hz, _, _ = protocols.drive(c, name)
            driven = hz > 0
            act = (full > 0.5) | (got > 0.5)
            if act.sum() < 3:
                continue
            js.append(((full > 0.5) & (got > 0.5)).sum() / act.sum())
            rs.append(np.corrcoef(full[act], got[act])[0, 1])
            dn = (~driven) & act
            if dn.sum() > 2:
                ratios.append(got[dn].sum() / max(full[dn].sum(), 1e-9))
                rdns.append(np.corrcoef(full[dn], got[dn])[0, 1])
        row = (k, np.mean(js), np.mean(rs), np.mean(ratios), np.mean(rdns))
        results.append(row)
        print(f"{k:4.2f}   {row[1]:.3f}     {row[2]:.3f}    {row[3]:8.3f}"
              f"          {row[4]:.3f}")

    # prefer the k whose downstream firing best matches, then correlation
    best = min(results, key=lambda r: abs(np.log(max(r[3], 1e-6))) - 0.3 * r[4])
    print(f"\nchosen k = {best[0]:.2f}: Jaccard {best[1]:.3f}, rate r {best[2]:.3f}, "
          f"downstream ratio {best[3]:.3f}, downstream r {best[4]:.3f}")
    with open(ROOT / "build/bias_fit.json", "w") as f:
        json.dump({"k": best[0], "jaccard": best[1], "rate_r": best[2],
                   "downstream_ratio": best[3], "downstream_r": best[4],
                   "sweep": [list(map(float, r)) for r in results]}, f, indent=2)
    print("wrote build/bias_fit.json")


if __name__ == "__main__":
    main()
