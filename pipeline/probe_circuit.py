"""Sanity-check the extracted sub-connectome: does it still behave like a fly?

Three experiments, each the in-silico version of a published fly experiment:

  1. quiet      -- no input at all; the network must fall silent (the Shiu model
                   has no background drive, so any activity here is runaway gain)
  2. looming    -- drive right-eye LC4 + LPLC2 as an approaching object;
                   expect DNp09/DNp01 escape neurons to fire and the descending
                   population to become left/right asymmetric
  3. compass    -- drive EPG, then drive PEN_a asymmetrically; expect a
                   localised bump of EPG activity that moves when PEN drive is
                   made asymmetric (the ring attractor integrating turning)
"""
from __future__ import annotations

import numpy as np

from lif import LIFNetwork
from netio import Circuit


def summarise(c, res, label, top=12):
    r = res["rates_hz"]
    act = np.flatnonzero(r > 0.5)
    print(f"\n--- {label}: {len(act)} neurons active, mean {r[act].mean() if len(act) else 0:.1f} Hz")
    order = np.argsort(-r)[:top]
    for i in order:
        if r[i] <= 0.5:
            break
        print(f"    {c.label(i):22s} {r[i]:7.1f} Hz  ({c.role[i]})")
    return r


def group_rate(c, r, **kw):
    ids = c.ids(**kw)
    return r[ids].mean() if len(ids) else 0.0


def main():
    c = Circuit()
    print(f"circuit: {c.N} neurons, {c.W.nnz} edges")
    net = LIFNetwork(c.W, dt=1.0, seed=1)

    # ---------------------------------------------------------------- quiet
    net.reset()
    res = net.run(500.0)
    r0 = summarise(c, res, "quiet (no input)")
    assert r0.max() < 1.0, "network self-excites with no input -- gain is wrong"

    # -------------------------------------------------------------- looming
    # An object looming on the fly's right: right-eye LC4/LPLC2 whose azimuth is
    # near the front-right are driven hard, the rest weakly.
    loom_hz = np.zeros(c.N)
    for t in ("LC4", "LPLC2", "LC6", "LPLC1"):
        ids = c.ids(group=t, side="right")
        # gaussian tuning around +45 deg azimuth
        loom_hz[ids] = 150.0 * np.exp(-((c.az[ids] - 45.0) ** 2) / (2 * 35.0**2))
    drive = c.ids(role=["looming", "small_object", "bar", "optic_flow"])
    no_rfc = np.zeros(c.N, bool)
    no_rfc[np.flatnonzero(loom_hz > 1)] = True

    net.reset()
    res = net.run(500.0, rate_fn=lambda t: loom_hz, no_refractory=no_rfc)
    r1 = summarise(c, res, "looming from the right")
    print(f"    escape DNp09/DNp01 : {group_rate(c, r1, type_=['DNp09','DNp01','DNp10','DNp11']):.1f} Hz")
    print(f"    DN left            : {group_rate(c, r1, role='descending', side='left'):.1f} Hz")
    print(f"    DN right           : {group_rate(c, r1, role='descending', side='right'):.1f} Hz")
    print(f"    DNa02 left/right   : {group_rate(c, r1, type_='DNa02', side='left'):.1f} / "
          f"{group_rate(c, r1, type_='DNa02', side='right'):.1f} Hz")

    # -------------------------------------------------------------- compass
    epg = c.ids(type_="EPG")
    pen_l = c.ids(type_="PEN_a(PEN1)", side="left")
    pen_r = c.ids(type_="PEN_a(PEN1)", side="right")
    seed_epg = epg[: max(3, len(epg) // 8)]

    hz = np.zeros(c.N)
    hz[seed_epg] = 120.0
    no_rfc = np.zeros(c.N, bool)
    no_rfc[seed_epg] = True
    net.reset()
    res = net.run(400.0, rate_fn=lambda t: hz, no_refractory=no_rfc)
    r2 = summarise(c, res, "compass: seeding one EPG wedge", top=10)
    epg_r = r2[epg]
    print(f"    EPG active {np.sum(epg_r>1)}/{len(epg)}  max {epg_r.max():.0f} Hz  "
          f"mean {epg_r.mean():.1f} Hz   (a bump should be sparse, not all-on)")
    print(f"    Delta7 (inhibition): {group_rate(c, r2, type_='Delta7'):.1f} Hz")
    print(f"    PEN_a  {group_rate(c, r2, type_='PEN_a(PEN1)'):.1f} Hz   "
          f"PFL3 {group_rate(c, r2, type_='PFL3'):.1f} Hz")


if __name__ == "__main__":
    main()
