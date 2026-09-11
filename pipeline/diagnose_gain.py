"""Where does the sub-network's 1.3x excess firing come from?

Splits the sub-vs-full comparison by role and by whether a neuron is directly
Poisson-driven, so the gain correction is applied to the right thing instead of
being fitted blind.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import protocols
from calibrate_gain import full_brain_rates, sub_rates
from netio import Circuit, ROOT


def main():
    c = Circuit()
    W = sp.load_npz(ROOT / "build/W_pruned.npz")
    ref = full_brain_rates(c)

    for name in ["loom_right", "landmark_front"]:
        hz, _, _ = protocols.drive(c, name)
        driven = hz > 0
        got = sub_rates(c, W, name, 1.0, trials=4)
        r = ref[name]
        print(f"\n=== {name} ===")
        for label, m in [("Poisson-driven", driven), ("downstream", ~driven)]:
            act = m & ((r > 0.5) | (got > 0.5))
            if act.sum() < 2:
                continue
            print(f"  {label:15s} n={act.sum():4d}  full {r[act].sum():8.0f} Hz  "
                  f"sub {got[act].sum():8.0f} Hz  ratio {got[act].sum()/max(r[act].sum(),1e-9):.3f}"
                  f"  r={np.corrcoef(r[act], got[act])[0,1]:.3f}")
        print("  by role (downstream only):")
        for role in sorted(set(c.role)):
            m = (~driven) & (c.role == role) & ((r > 0.5) | (got > 0.5))
            if m.sum() < 3:
                continue
            print(f"    {role:16s} n={m.sum():4d}  full {r[m].sum():7.0f}  sub {got[m].sum():7.0f}"
                  f"  ratio {got[m].sum()/max(r[m].sum(),1e-9):6.3f}")


if __name__ == "__main__":
    main()
