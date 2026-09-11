"""Why doesn't the compass sustain?  Measure how much real input each central
complex population lost to truncation, and how strong the surviving loops are."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from netio import Circuit, ROOT


def main():
    c = Circuit()
    with open(ROOT / "build/circuit.json") as f:
        raw = json.load(f)
    cover = np.array([n["in_cover"] for n in raw["neurons"]])

    print("input coverage (fraction of each neuron's real synaptic input kept):")
    for role in sorted(set(c.role)):
        m = c.role == role
        print(f"  {role:16s} n={m.sum():4d}  median {np.median(cover[m]):.2f}  "
              f"p10 {np.percentile(cover[m],10):.2f}  p90 {np.percentile(cover[m],90):.2f}")

    print("\nkey populations:")
    W = c.W
    pos_in = np.asarray(W.multiply(W > 0).sum(axis=0)).ravel()
    neg_in = np.asarray(W.multiply(W < 0).sum(axis=0)).ravel()
    for t in ["EPG", "PEN_a(PEN1)", "PEN_b(PEN2)", "Delta7", "PEG", "PFL3", "PFL2",
              "DNa02", "DNa01", "DNp09", "DNp01"]:
        ids = c.ids(type_=t)
        if not len(ids):
            continue
        print(f"  {t:14s} n={len(ids):3d}  cover {np.median(cover[ids]):.2f}  "
              f"exc-in {pos_in[ids].mean():7.1f} mV  inh-in {neg_in[ids].mean():7.1f} mV  "
              f"(threshold is 7 mV above rest)")

    print("\nloop gains inside the heading system (total mV per population):")
    def blk(a, b):
        ia, ib = c.ids(type_=a), c.ids(type_=b)
        if not len(ia) or not len(ib):
            return 0.0, 0
        sub = W[ia][:, ib]
        return float(sub.sum()), sub.nnz
    for a, b in [("EPG", "PEN_a(PEN1)"), ("PEN_a(PEN1)", "EPG"),
                 ("EPG", "PEN_b(PEN2)"), ("PEN_b(PEN2)", "EPG"),
                 ("EPG", "Delta7"), ("Delta7", "EPG"), ("Delta7", "PEN_a(PEN1)"),
                 ("EPG", "PEG"), ("PEG", "EPG"),
                 ("EPG", "PFL3"), ("PFL3", "DNa02"), ("PFL2", "DNa02")]:
        s, nnz = blk(a, b)
        print(f"  {a:14s} -> {b:14s} {s:9.1f} mV over {nnz:5d} edges")

    print("\nwhat is missing from PFL3 (top lost presynaptic types)?")
    # Recompute lost input for PFL3 from the full connectome.
    import pandas as pd, pyarrow.parquet as pq
    comp = pd.read_csv(ROOT / "fly-brain/data/2025_Completeness_783.csv", index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    gidx = {int(r): i for i, r in enumerate(root_ids)}
    ann = pd.read_csv(ROOT / "data/flywire_annotations.tsv", sep="\t", low_memory=False)
    ann = ann[ann.root_id.isin(gidx)]
    ann_type = dict(zip(ann.root_id.map(gidx), ann.cell_type))

    t = pq.read_table(ROOT / "fly-brain/data/2025_Connectivity_783.parquet",
                      columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity"])
    pre, post, n = (t.column(i).to_numpy() for i in range(3))
    kept_global = set(int(x) for x in
                      pd.Series(root_ids).reset_index().set_index(0)
                      .loc[c.root_id, "index"].to_numpy())

    for target in ["PFL3", "PEN_a(PEN1)", "EPG"]:
        tgt_global = {int(gidx[r]) for r in c.root_id[c.ids(type_=target)]}
        m = np.isin(post, list(tgt_global))
        lost = m & ~np.isin(pre, list(kept_global))
        agg = {}
        for p, k in zip(pre[lost], n[lost]):
            agg[ann_type.get(int(p), "?")] = agg.get(ann_type.get(int(p), "?"), 0) + int(k)
        top = sorted(agg.items(), key=lambda x: -x[1])[:8]
        print(f"  {target:14s} lost {sum(agg.values()):7d} syn; top: "
              + ", ".join(f"{k}:{v}" for k, v in top))


if __name__ == "__main__":
    main()
