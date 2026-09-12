"""Do smell and heat reach the motor output at all in this model?

Before extracting an olfactory or thermosensory pathway into the mod, ask the
whole 138 639-neuron brain whether the signal gets anywhere near a descending
neuron.  This is the same discipline that caught the recruit experiment: find
out before building.

Protocols, each driving real olfactory receptor neurons or thermoreceptors the
way the paper drives optogenetic activation:

  odour_attract     ORN_DM1 (Or42b) + ORN_DM4 (Or59b), both antennae.
                    Apple cider vinegar / fermentation.  The classic
                    attractive pair (Semmelhack & Wang 2009).
  odour_right       the same, right antenna only -- does the brain produce a
                    left/right difference a fly could steer on?
  odour_averse      ORN_DA2 (Or56a, geosmin -- the smell of harmful microbes,
                    innately aversive; Stensmyr 2012) + ORN_V (CO2).
  heat              TRN_VP1m / VP2 / VP3, the antennal thermoreceptors.

For each we report how far the signal travels: receptors -> projection neurons
-> lateral horn -> descending neurons, plus which DNs respond and whether they
are lateralised.

Run:  python pipeline/probe_new_senses.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from lif import LIFNetwork
from netio import ROOT

DURATION_MS = 1000.0
TRIALS = 2
RATE_HZ = 150.0


def load_full():
    comp = pd.read_csv(ROOT / "fly-brain/data/2025_Completeness_783.csv", index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    N = len(root_ids)
    pos = pd.Series(np.arange(N), index=root_ids)

    ann = pd.read_csv(ROOT / "data/flywire_annotations.tsv", sep="\t", low_memory=False)
    ann = ann[ann.root_id.isin(pos.index)].copy()
    ann["idx"] = pos.reindex(ann.root_id).to_numpy()
    ann["cell_type"] = ann["cell_type"].astype("string")

    print("loading the full connectome (15.1M edges)...")
    t = pq.read_table(ROOT / "fly-brain/data/2025_Connectivity_783.parquet",
                      columns=["Presynaptic_Index", "Postsynaptic_Index",
                               "Excitatory x Connectivity"])
    W = sp.csr_matrix((t.column(2).to_numpy().astype(np.float32) * 0.275,
                       (t.column(0).to_numpy(), t.column(1).to_numpy())), shape=(N, N))
    del t
    return N, ann, W


def idx_of_types(ann, types, side=None):
    m = ann.cell_type.isin(types)
    if side:
        m &= ann.side == side
    return ann[m].idx.to_numpy()


def idx_prefix(ann, prefixes, super_class=None):
    m = ann.cell_type.str.startswith(tuple(prefixes), na=False)
    if super_class:
        m &= ann.super_class == super_class
    return ann[m].idx.to_numpy()


def main():
    N, ann, W = load_full()

    ATTRACT = ["ORN_DM1", "ORN_DM4", "ORN_DM2", "ORN_VM2"]
    AVERSE = ["ORN_DA2", "ORN_V"]
    THERMO = ["TRN_VP1m", "TRN_VP2", "TRN_VP3a", "TRN_VP3b"]

    protocols = {
        "odour_attract": idx_of_types(ann, ATTRACT),
        "odour_right": idx_of_types(ann, ATTRACT, side="right"),
        "odour_averse": idx_of_types(ann, AVERSE),
        "heat": idx_of_types(ann, THERMO),
    }

    # where the signal has to travel through
    stages = {
        "ORN (receptors)": idx_prefix(ann, ["ORN_"]),
        "TRN/HRN (thermo/hygro)": idx_prefix(ann, ["TRN_", "HRN_"]),
        "PN (antennal lobe out)": ann[ann.cell_type.str.contains("PN", na=False)
                                      & (ann.super_class == "central")].idx.to_numpy(),
        "lateral horn": idx_prefix(ann, ["LHAV", "LHPV", "LHPD", "LHAD"]),
        "Kenyon cells": idx_prefix(ann, ["KC"]),
        "descending": ann[ann.super_class == "descending"].idx.to_numpy(),
    }
    dn_idx = stages["descending"]
    dn_type = ann.set_index("idx").reindex(dn_idx)["cell_type"].fillna("?").to_numpy()
    dn_side = ann.set_index("idx").reindex(dn_idx)["side"].fillna("?").to_numpy()

    for name, drive_idx in protocols.items():
        hz = np.zeros(N)
        hz[drive_idx] = RATE_HZ
        no_rfc = hz > 0
        acc = np.zeros(N)
        for tr in range(TRIALS):
            net = LIFNetwork(W, dt=1.0, seed=500 + tr)
            acc += net.run(DURATION_MS, rate_fn=lambda _t: hz,
                           no_refractory=no_rfc, record=False)["counts"]
        rates = acc / TRIALS / (DURATION_MS / 1000.0)

        print(f"\n=== {name}: driving {len(drive_idx)} receptors at {RATE_HZ:.0f} Hz ===")
        print(f"    {(rates > 0.5).sum()} neurons active in the whole brain")
        for stage, ids in stages.items():
            if len(ids) == 0:
                continue
            r = rates[ids]
            print(f"    {stage:24s} {int((r > 0.5).sum()):5d}/{len(ids):5d} active, "
                  f"mean {r.mean():6.2f} Hz, max {r.max():6.1f} Hz")

        r_dn = rates[dn_idx]
        hot = np.argsort(-r_dn)[:10]
        hot = [h for h in hot if r_dn[h] > 0.5]
        if hot:
            print("    top descending neurons: " + ", ".join(
                f"{dn_type[h]}[{dn_side[h][:1]}] {r_dn[h]:.0f}Hz" for h in hot))
            left = r_dn[dn_side == "left"].mean()
            right = r_dn[dn_side == "right"].mean()
            print(f"    DN left {left:.2f} Hz   DN right {right:.2f} Hz   "
                  f"asymmetry {right - left:+.2f} Hz")
        else:
            print("    NO descending neuron responds -- this sense cannot steer "
                  "the fly in this model")


if __name__ == "__main__":
    main()
