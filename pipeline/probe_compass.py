"""Does the central complex work in this model?  Measured answer: no, and why.

The circuit contains the fly's whole heading system -- the MeTu -> TuBu -> ER
landmark pathway, the EPG/PEN/PEG/Delta7 ring attractor, and the PFL2/PFL3
steering output -- 948 of its 2053 neurons.  This script establishes three
things, and the mod's documentation reports all three:

  1. The landmark pathway is intact and azimuth-selective as far as the ring
     neurons: where a landmark sits in the visual field measurably changes
     which ER neurons fire.
  2. The EPG ring attractor does not form a bump.  With no input it is silent;
     with uniform drive it saturates uniformly.  ER -> EPG is purely
     inhibitory, so the landmark can only carve a hole in excitation that the
     model never supplies.
  3. Consequently PFL3 -- the steering output -- carries no heading signal,
     and the mod does not use one.

This is a property of the Shiu et al. model, not of the extraction: that model
deliberately starts from total silence to model optogenetic activation, and a
ring attractor cannot bootstrap itself from silence.

Run:  python pipeline/probe_compass.py
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from lif import LIFNetwork
from netio import Circuit, ROOT

AZIMUTHS = [-150, -90, -45, 0, 45, 90, 150]


def run(c, W, bias, epg, tonic, az, ms=1500, seed=5):
    hz = np.zeros(c.N)
    for i in c.ids(role="landmark"):
        d = (c.az[i] - az) / 25.0
        r = 120.0 * np.exp(-0.5 * d * d)
        if r > 1.0:
            hz[i] = r
    if tonic:
        hz[epg] += tonic
    net = LIFNetwork(W, dt=1.0, seed=seed, bias=bias)
    return net.run(ms, rate_fn=lambda _t: hz, no_refractory=hz > 0,
                   record=False)["rates_hz"]


def similarity(patterns):
    P = np.array(patterns)
    if P.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
    sim = Pn @ Pn.T
    nrm = np.sqrt(np.diag(sim))
    cos = sim / np.outer(nrm, nrm)
    n = len(patterns)
    return (cos[np.triu_indices(n, 1)].mean(),
            np.mean([cos[i, i + 1] for i in range(n - 1)]),
            cos[0, -1])


def main():
    c = Circuit()
    W = sp.load_npz(ROOT / "build/W_pruned.npz")
    bias = c.bias()
    epg = c.ids(type_="EPG")
    er = np.array([i for i in range(c.N) if c.type[i].startswith("ER")])
    tubu = np.array([i for i in range(c.N) if c.type[i].startswith("TuBu")])

    print("ER -> EPG total weight: "
          f"{W[er][:, epg].sum():.0f} mV over {W[er][:, epg].nnz} edges "
          "(ring neurons are GABAergic: this pathway can only subtract)\n")

    print("1 + 3. landmark azimuth -> ring neurons -> steering output")
    print("   az   TuBu on   ER on   EPG L/R (Hz)   PFL3 L/R   PFL2 L/R")
    er_pat, epg_pat = [], []
    for az in AZIMUTHS:
        r = run(c, W, bias, epg, 90, az)
        er_pat.append(r[er])
        epg_pat.append(r[epg])

        def m(t, side=None):
            ids = c.ids(type_=t, side=side)
            return r[ids].mean() if len(ids) else 0.0

        epgL = r[[i for i in epg if c.side[i] == "left"]].mean()
        epgR = r[[i for i in epg if c.side[i] == "right"]].mean()
        print(f" {az:+5d}   {int((r[tubu] > 0.5).sum()):3d}/{len(tubu)}  "
              f"{int((r[er] > 0.5).sum()):3d}/{len(er)}   {epgL:5.1f}/{epgR:5.1f}"
              f"      {m('PFL3','left'):4.1f}/{m('PFL3','right'):4.1f}"
              f"    {m('PFL2','left'):4.1f}/{m('PFL2','right'):4.1f}")

    mean_er, adj_er, opp_er = similarity(er_pat)
    mean_epg, adj_epg, opp_epg = similarity(epg_pat)
    print(f"\n  ring-neuron pattern similarity: adjacent azimuths {adj_er:.3f}, "
          f"opposite {opp_er:.3f}")
    print("  -> the landmark's direction does reach the ring neurons")
    print(f"  EPG pattern similarity:         adjacent azimuths {adj_epg:.3f}, "
          f"opposite {opp_epg:.3f}")
    print("  -> but the compass itself does not distinguish them")

    print("\n2. can the ring attractor be switched on at all?")
    print("  tonic EPG drive   EPG active/  n   EPG mean   PEN    Delta7   PFL3")
    for tonic in [0, 15, 40, 90]:
        r = run(c, W, bias, epg, tonic, 0)
        e = r[epg]

        def m(t):
            ids = c.ids(type_=t)
            return r[ids].mean() if len(ids) else 0.0

        print(f"  {tonic:6.0f} Hz          {int((e > 0.5).sum()):3d}/{len(epg):3d}"
              f"      {e.mean():7.1f}   {m('PEN_a(PEN1)'):5.1f}  {m('Delta7'):6.1f}"
              f"  {m('PFL3'):5.1f}")
    print("\n  A bump would be a few EPG neurons firing while the rest stay quiet.")
    print("  Instead the population goes from all-off to all-on: no attractor.")


if __name__ == "__main__":
    main()
