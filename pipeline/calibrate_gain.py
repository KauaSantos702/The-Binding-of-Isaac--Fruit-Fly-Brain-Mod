"""Correct the sub-network's gain against the whole 138 639-neuron model.

Truncating the brain to 2053 neurons removes ~60% of each neuron's real input.
Excitation and inhibition are not removed in equal measure, so the extracted
circuit responds about 30% more strongly than the full model does to the same
stimulus (see validate.py part C).  This script finds the single scalar gain on
w_syn that best restores agreement, and reports the residual error.

One free parameter, fitted against the full model -- not per-synapse tuning.

Run:  python pipeline/calibrate_gain.py
Writes build/gain.json, consumed by export_lua.py.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

import protocols
from lif import LIFNetwork
from netio import Circuit, ROOT

CAL_PROTOCOLS = ["loom_right", "loom_left", "landmark_front", "loom_front"]
TRIALS_FULL = 3
TRIALS_SUB = 6
GAINS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]


def full_brain_rates(circuit):
    """Cached per-protocol firing rates of the whole brain, restricted to ours."""
    cache = ROOT / "build/full_rates.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    import pyarrow.parquet as pq

    comp = pd.read_csv(ROOT / "fly-brain/data/2025_Completeness_783.csv", index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    N = len(root_ids)
    pos_of = pd.Series(np.arange(N), index=root_ids)
    gidx = pos_of.reindex(circuit.root_id).to_numpy()

    print("loading the full connectome (15.1M edges)...")
    t = pq.read_table(ROOT / "fly-brain/data/2025_Connectivity_783.parquet",
                      columns=["Presynaptic_Index", "Postsynaptic_Index",
                               "Excitatory x Connectivity"])
    W = sp.csr_matrix((t.column(2).to_numpy().astype(np.float32) * 0.275,
                       (t.column(0).to_numpy(), t.column(1).to_numpy())), shape=(N, N))
    del t

    out = {}
    for name in CAL_PROTOCOLS:
        hz_sub, _, ms = protocols.drive(circuit, name)
        hz = np.zeros(N)
        hz[gidx] = hz_sub
        no_rfc = hz > 0
        acc = np.zeros(N)
        for tr in range(TRIALS_FULL):
            net = LIFNetwork(W, dt=1.0, seed=100 + tr)
            acc += net.run(ms, rate_fn=lambda _t: hz, no_refractory=no_rfc,
                           record=False)["counts"]
        out[name] = (acc / TRIALS_FULL / (ms / 1000.0))[gidx]
        print(f"  {name}: {(out[name] > 0.5).sum()} of our neurons active in the full model")
    np.savez(cache, **out)
    return out


def sub_rates(circuit, W, name, gain, trials=TRIALS_SUB):
    hz, no_rfc, ms = protocols.drive(circuit, name)
    Wg = W * gain
    acc = np.zeros(W.shape[0])
    for tr in range(trials):
        net = LIFNetwork(Wg, dt=1.0, seed=200 + tr)
        acc += net.run(ms, rate_fn=lambda _t: hz, no_refractory=no_rfc,
                       record=False)["counts"]
    return acc / trials / (ms / 1000.0)


def score(ref, got):
    both = (ref > 0.5) | (got > 0.5)
    if both.sum() < 3:
        return 0.0, 0.0, 0.0
    jac = ((ref > 0.5) & (got > 0.5)).sum() / both.sum()
    r = np.corrcoef(ref[both], got[both])[0, 1]
    ratio = got[both].sum() / max(ref[both].sum(), 1e-9)
    return jac, r, ratio


def main():
    circuit = Circuit()
    W = sp.load_npz(ROOT / "build/W_pruned.npz")
    ref = full_brain_rates(circuit)

    print("\ngain   Jaccard   rate r   rate ratio (sub/full)")
    results = []
    for g in GAINS:
        js, rs, ra = [], [], []
        for name in CAL_PROTOCOLS:
            j, r, ratio = score(ref[name], sub_rates(circuit, W, name, g))
            js.append(j); rs.append(r); ra.append(ratio)
        j, r, ratio = np.mean(js), np.mean(rs), np.mean(ra)
        results.append((g, j, r, ratio))
        print(f"{g:4.2f}   {j:.3f}     {r:.4f}   {ratio:.3f}")

    # pick the gain whose total rate ratio is closest to 1, breaking ties on r
    best = min(results, key=lambda x: (abs(x[3] - 1.0) - 0.25 * x[2]))
    print(f"\nchosen gain {best[0]:.2f}  (Jaccard {best[1]:.3f}, r {best[2]:.4f}, "
          f"rate ratio {best[3]:.3f})")
    with open(ROOT / "build/gain.json", "w") as f:
        json.dump({"gain": best[0], "jaccard": best[1], "rate_r": best[2],
                   "rate_ratio": best[3],
                   "note": "single scalar on w_syn, fitted so the 2053-neuron "
                           "circuit matches the whole 138639-neuron model's "
                           "response to the same stimuli"}, f, indent=2)
    print("wrote build/gain.json")


if __name__ == "__main__":
    main()
