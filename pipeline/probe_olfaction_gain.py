"""Is the olfactory blow-up the model, or did we just drive it too hard?

probe_new_senses.py found that driving olfactory receptors at the paper's
standard 150 Hz makes ~10 000 neurons fire and produces the *same* descending
output whatever you stimulate -- vinegar, geosmin, heat, one antenna or two.
That is a saturated brain, not a stimulus response.

This sweeps the drive rate down.  If specificity is there at lower rates, the
pathway is usable and we build it; if the response stays identical all the way
down, smell cannot steer the fly in this model and the mod must not pretend it
does.

Specificity is measured as the cosine distance between the descending-neuron
response vectors of different stimuli.  Identical responses give a similarity
of 1.0; a brain that distinguishes vinegar from geosmin gives clearly less.

Run:  python pipeline/probe_olfaction_gain.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from lif import LIFNetwork
from netio import ROOT

DURATION_MS = 800.0
RATES = [10.0, 25.0, 50.0, 100.0, 150.0]

ATTRACT = ["ORN_DM1", "ORN_DM4", "ORN_DM2", "ORN_VM2"]
AVERSE = ["ORN_DA2", "ORN_V"]
THERMO = ["TRN_VP1m", "TRN_VP2", "TRN_VP3a", "TRN_VP3b"]


def main():
    comp = pd.read_csv(ROOT / "fly-brain/data/2025_Completeness_783.csv", index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    N = len(root_ids)
    pos = pd.Series(np.arange(N), index=root_ids)

    ann = pd.read_csv(ROOT / "data/flywire_annotations.tsv", sep="\t", low_memory=False)
    ann = ann[ann.root_id.isin(pos.index)].copy()
    ann["idx"] = pos.reindex(ann.root_id).to_numpy()
    ann["cell_type"] = ann["cell_type"].astype("string")

    print("loading the full connectome...")
    t = pq.read_table(ROOT / "fly-brain/data/2025_Connectivity_783.parquet",
                      columns=["Presynaptic_Index", "Postsynaptic_Index",
                               "Excitatory x Connectivity"])
    W = sp.csr_matrix((t.column(2).to_numpy().astype(np.float32) * 0.275,
                       (t.column(0).to_numpy(), t.column(1).to_numpy())), shape=(N, N))
    del t

    def sel(types, side=None):
        m = ann.cell_type.isin(types)
        if side:
            m &= ann.side == side
        return ann[m].idx.to_numpy()

    stimuli = {
        "vinegar": sel(ATTRACT),
        "vinegar-R": sel(ATTRACT, "right"),
        "geosmin": sel(AVERSE),
        "heat": sel(THERMO),
    }
    dn = ann[ann.super_class == "descending"].idx.to_numpy()
    dn_side = ann.set_index("idx").reindex(dn)["side"].fillna("?").to_numpy()

    def run(idx, rate):
        hz = np.zeros(N)
        hz[idx] = rate
        net = LIFNetwork(W, dt=1.0, seed=909)
        return net.run(DURATION_MS, rate_fn=lambda _t: hz,
                       no_refractory=hz > 0, record=False)["counts"] / (DURATION_MS / 1000.0)

    print("\nrate   whole-brain active   DN active   vinegar-vs-geosmin   "
          "vinegar-vs-heat   both-vs-right-antenna   DN L/R asymmetry")
    for rate in RATES:
        resp = {k: run(v, rate) for k, v in stimuli.items()}
        dns = {k: v[dn] for k, v in resp.items()}

        def cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                return float("nan")
            return float(a @ b / (na * nb))

        act = int((resp["vinegar"] > 0.5).sum())
        dn_act = int((dns["vinegar"] > 0.5).sum())
        vg = cos(dns["vinegar"], dns["geosmin"])
        vh = cos(dns["vinegar"], dns["heat"])
        vr = cos(dns["vinegar"], dns["vinegar-R"])
        d = dns["vinegar"]
        asym = d[dn_side == "right"].mean() - d[dn_side == "left"].mean()
        print(f"{rate:5.0f}  {act:16d}  {dn_act:10d}   {vg:17.4f} {vh:17.4f} "
              f"{vr:22.4f}   {asym:+.3f} Hz")

    print("\n  similarity 1.000 = the brain gives the same motor output whatever it smells")
    print("  a usable olfactory sense needs these to fall clearly below 1")


if __name__ == "__main__":
    main()
