"""Find the neurons the *whole* brain recruits for our stimuli but that the
extracted circuit is missing, so they can be added back.

The sub-network fires ~2.5x too hard downstream because the full brain also
activates ~700 neurons outside our selection, most of them inhibitory feedback
onto exactly the populations that are running hot.  Rather than fudge a gain,
we run the full 138 639-neuron model on every protocol, rank the neurons it
activates, and hand the strongest missing ones back to build_circuit.py.

Run:  python pipeline/find_recruits.py [--top 800]
Writes build/recruits.json (a list of FlyWire root ids) and
build/full_rates_all.npz (whole-brain rates per protocol, cached).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

import protocols
from lif import LIFNetwork
from netio import Circuit, ROOT

TRIALS = 2


def whole_brain_rates(circuit):
    cache = ROOT / "build/full_rates_all.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files if k != "root_ids"}, d["root_ids"]

    import pyarrow.parquet as pq

    comp = pd.read_csv(ROOT / "fly-brain/data/2025_Completeness_783.csv", index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    N = len(root_ids)
    gidx = pd.Series(np.arange(N), index=root_ids).reindex(circuit.root_id).to_numpy()

    print("loading the full connectome (15.1M edges)...")
    t = pq.read_table(ROOT / "fly-brain/data/2025_Connectivity_783.parquet",
                      columns=["Presynaptic_Index", "Postsynaptic_Index",
                               "Excitatory x Connectivity"])
    W = sp.csr_matrix((t.column(2).to_numpy().astype(np.float32) * 0.275,
                       (t.column(0).to_numpy(), t.column(1).to_numpy())), shape=(N, N))
    del t

    out = {}
    for name in protocols.PROTOCOLS:
        hz_sub, _, ms = protocols.drive(circuit, name)
        hz = np.zeros(N)
        hz[gidx] = hz_sub
        no_rfc = hz > 0
        acc = np.zeros(N)
        for tr in range(TRIALS):
            net = LIFNetwork(W, dt=1.0, seed=300 + tr)
            acc += net.run(ms, rate_fn=lambda _t: hz, no_refractory=no_rfc,
                           record=False)["counts"]
        out[name] = acc / TRIALS / (ms / 1000.0)
        print(f"  {name}: {(out[name] > 0.5).sum()} neurons active in the whole brain")
    np.savez(cache, root_ids=root_ids, **out)
    return out, root_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=800)
    ap.add_argument("--min-hz", type=float, default=1.0)
    args = ap.parse_args()

    c = Circuit()
    rates, root_ids = whole_brain_rates(c)

    have = set(int(x) for x in c.root_id)
    # rank a neuron by its best response across protocols
    best = np.zeros(len(root_ids))
    for v in rates.values():
        best = np.maximum(best, v)

    order = np.argsort(-best)
    picked = []
    for k in order:
        rid = int(root_ids[k])
        if rid in have:
            continue
        if best[k] < args.min_hz:
            break
        picked.append(rid)
        if len(picked) >= args.top:
            break

    print(f"\n{len(picked)} recruits (>= {args.min_hz} Hz in the whole brain, "
          f"not already in the circuit)")

    ann = pd.read_csv(ROOT / "data/flywire_annotations.tsv", sep="\t", low_memory=False)
    ann = ann.set_index("root_id")
    types = ann.reindex(picked)["cell_type"].fillna("?")
    sup = ann.reindex(picked)["super_class"].fillna("?")
    print("  super classes:", sup.value_counts().head(8).to_dict())
    print("  top types    :", types.value_counts().head(15).to_dict())

    with open(ROOT / "build/recruits.json", "w") as f:
        json.dump({"root_ids": picked, "min_hz": args.min_hz}, f)
    print("wrote build/recruits.json")


if __name__ == "__main__":
    main()
