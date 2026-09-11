"""Three-way validation of the mod's brain.

  A. Lua engine  vs  numpy reference     -- same network, same protocols.
                                            Different RNG streams, so we
                                            compare per-neuron firing rates,
                                            the same measure fly-brain's own
                                            compare_ground_truth.py uses.
  B. pruned      vs  unpruned network    -- does dropping 1-synapse
                                            connections change behaviour?
  C. sub-network vs  whole 138k brain    -- does the extracted circuit still
                                            respond like the full model?

Run:  python pipeline/validate.py            (A and B)
      python pipeline/validate.py --full     (adds C; slow, needs ~4 GB)
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import protocols
from lif import LIFNetwork
from netio import Circuit, ROOT

TRIALS = 8


def run_reference(W, circuit, name, trials=TRIALS, seed0=0):
    hz, no_rfc, ms = protocols.drive(circuit, name)
    bias = circuit.bias()
    acc = np.zeros(W.shape[0])
    for t in range(trials):
        net = LIFNetwork(W, dt=1.0, seed=seed0 + t, bias=bias)
        res = net.run(ms, rate_fn=lambda _t: hz, no_refractory=no_rfc, record=False)
        acc += res["counts"]
    return acc / trials / (ms / 1000.0)


def compare(a, b, label_a, label_b, name):
    """Population comparison: active-set overlap + rate correlation."""
    act_a, act_b = a > 0.5, b > 0.5
    inter = (act_a & act_b).sum()
    union = (act_a | act_b).sum()
    jac = inter / union if union else 1.0
    both = act_a | act_b
    if both.sum() > 2:
        r = np.corrcoef(a[both], b[both])[0, 1]
    else:
        r = float("nan")
    ratio = b[both].sum() / max(a[both].sum(), 1e-9)
    print(f"  {name:16s} active {act_a.sum():4d}/{act_b.sum():4d}  "
          f"Jaccard {jac:.3f}  rate r {r:.4f}  total-rate {label_b}/{label_a} {ratio:.3f}")
    return jac, r, ratio


def part_a(circuit, W):
    print("\n=== A. Lua engine vs numpy reference (same network) ===")
    print("running the Lua harness...")
    lua = ROOT / "tools/lua53/lua53.exe"
    r = subprocess.run([str(lua), "tools/dump_rates.lua", str(TRIALS)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("lua harness failed")
    print("   ", r.stdout.strip().replace("\n", "\n    "))

    df = pd.read_csv(ROOT / "build/lua_rates.csv")
    rows = []
    for name in protocols.PROTOCOLS:
        ref = run_reference(W, circuit, name)
        lua_rates = np.zeros(circuit.N)
        sub = df[df.protocol == name]
        lua_rates[sub.neuron.to_numpy() - 1] = sub.rate_hz.to_numpy()
        rows.append((name,) + compare(ref, lua_rates, "py", "lua", name))
    js = np.mean([x[1] for x in rows])
    rs = np.mean([x[2] for x in rows])
    print(f"  mean Jaccard {js:.3f}, mean rate correlation {rs:.4f}")
    assert js > 0.9 and rs > 0.95, "Lua engine diverges from the reference"
    return rows


def part_b(circuit):
    print("\n=== B. pruned vs unpruned connectivity ===")
    full = circuit.W
    pruned = sp.load_npz(ROOT / "build/W_pruned.npz")
    print(f"  {full.nnz} -> {pruned.nnz} edges")
    for name in protocols.PROTOCOLS:
        a = run_reference(full, circuit, name)
        b = run_reference(pruned, circuit, name)
        compare(a, b, "full", "pruned", name)


def part_c(circuit):
    """Does the 2053-neuron circuit respond like the whole 138 639-neuron brain?"""
    print("\n=== C. sub-network vs whole 138k-neuron brain ===")
    import pyarrow.parquet as pq

    comp = pd.read_csv(ROOT / "fly-brain/data/2025_Completeness_783.csv", index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    N = len(root_ids)
    pos_of = pd.Series(np.arange(N), index=root_ids)

    print("  loading the full connectome...")
    t = pq.read_table(ROOT / "fly-brain/data/2025_Connectivity_783.parquet",
                      columns=["Presynaptic_Index", "Postsynaptic_Index",
                               "Excitatory x Connectivity"])
    pre = t.column(0).to_numpy()
    post = t.column(1).to_numpy()
    w = t.column(2).to_numpy().astype(np.float32) * 0.275
    Wfull = sp.csr_matrix((w, (pre, post)), shape=(N, N))
    del t, pre, post, w
    print(f"  {Wfull.nnz} edges, {Wfull.data.nbytes / 1e6:.0f} MB")

    gidx = pos_of.reindex(circuit.root_id).to_numpy()

    for name in ["loom_right", "loom_left", "landmark_front"]:
        hz_sub, no_rfc_sub, ms = protocols.drive(circuit, name)
        hz_full = np.zeros(N)
        hz_full[gidx] = hz_sub
        no_rfc_full = hz_full > 0

        acc = np.zeros(N)
        for tr in range(2):
            net = LIFNetwork(Wfull, dt=1.0, seed=tr)
            res = net.run(ms, rate_fn=lambda _t: hz_full,
                          no_refractory=no_rfc_full, record=False)
            acc += res["counts"]
        full_rates = acc / 2 / (ms / 1000.0)

        sub_rates = run_reference(sp.load_npz(ROOT / "build/W_pruned.npz"),
                                  circuit, name, trials=2)
        # compare only on the neurons the sub-network actually contains
        compare(full_rates[gidx], sub_rates, "138k", "sub", name)
        print(f"    (whole brain: {(full_rates > 0.5).sum()} neurons active overall)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run the 138k-neuron check")
    ap.add_argument("--only", choices=["a", "b", "c"])
    args = ap.parse_args()

    circuit = Circuit()
    W = sp.load_npz(ROOT / "build/W_pruned.npz")
    print(f"circuit: {circuit.N} neurons, pruned to {W.nnz} edges")

    if args.only in (None, "a"):
        part_a(circuit, W)
    if args.only in (None, "b"):
        part_b(circuit)
    if args.full or args.only == "c":
        part_c(circuit)


if __name__ == "__main__":
    main()
