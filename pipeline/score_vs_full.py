"""Score the current circuit against the cached whole-brain rates.

Fast enough to sweep circuit sizes: the expensive part (simulating all 138 639
neurons) is cached in build/full_rates_all.npz by find_recruits.py.

Run:  python pipeline/score_vs_full.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

import protocols
from fit_bias import sub_rates
from netio import Circuit, ROOT

TRIALS = 4


def main(circuit=None, W=None, verbose=True):
    c = circuit or Circuit()
    W = W if W is not None else sp.load_npz(ROOT / "build/W_pruned.npz")
    d = np.load(ROOT / "build/full_rates_all.npz")
    root_ids = d["root_ids"]
    pos = pd.Series(np.arange(len(root_ids)), index=root_ids)
    gidx = pos.reindex(c.root_id).to_numpy()

    hz_driven = {}
    rows = []
    for name in protocols.PROTOCOLS:
        full = d[name][gidx]
        got = sub_rates(c, W, name, c.bias(), trials=TRIALS)
        hz, _, _ = protocols.drive(c, name)
        driven = hz > 0
        hz_driven[name] = driven

        act = (full > 0.5) | (got > 0.5)
        jac = ((full > 0.5) & (got > 0.5)).sum() / max(act.sum(), 1)
        r = np.corrcoef(full[act], got[act])[0, 1] if act.sum() > 2 else float("nan")
        dn = (~driven) & act
        ratio_dn = got[dn].sum() / max(full[dn].sum(), 1e-9)
        r_dn = np.corrcoef(full[dn], got[dn])[0, 1] if dn.sum() > 2 else float("nan")
        rows.append((name, jac, r, ratio_dn, r_dn, int(act.sum())))
        if verbose:
            print(f"  {name:16s} Jaccard {jac:.3f}  r {r:.3f}   "
                  f"downstream: ratio {ratio_dn:6.3f}  r {r_dn:.3f}")

    jac = np.mean([x[1] for x in rows])
    r = np.mean([x[2] for x in rows])
    ratio = np.mean([x[3] for x in rows])
    r_dn = np.mean([x[4] for x in rows])
    if verbose:
        print(f"  MEAN             Jaccard {jac:.3f}  r {r:.3f}   "
              f"downstream: ratio {ratio:6.3f}  r {r_dn:.3f}")
    return jac, r, ratio, r_dn


if __name__ == "__main__":
    c = Circuit()
    W = sp.load_npz(ROOT / "build/W_pruned.npz")
    print(f"circuit: {c.N} neurons, {W.nnz} edges (pruned)")
    main(c, W)
