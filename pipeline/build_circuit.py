"""Extract the mod's sub-connectome from the full FlyWire v783 network.

Inputs
    fly-brain/data/2025_Completeness_783.csv      neuron list (138 639 rows)
    fly-brain/data/2025_Connectivity_783.parquet  15 091 983 signed edges
    data/flywire_annotations.tsv                  cell types (Schlegel et al. 2024)

Output
    build/circuit.json   neurons + CSR connectivity + sensory maps + read-outs

Nothing about the connectivity is synthesised.  Selection and population
subsampling are the only operations applied; kept presynaptic neurons have their
outgoing weights rescaled by n_total/n_kept so that downstream drive is
preserved in expectation.

Usage:  python pipeline/build_circuit.py [--out build/circuit.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

import circuit_spec as spec

ROOT = Path(__file__).resolve().parent.parent
COMP = ROOT / "fly-brain/data/2025_Completeness_783.csv"
CONN = ROOT / "fly-brain/data/2025_Connectivity_783.parquet"
ANN = ROOT / "data/flywire_annotations.tsv"

# FlyWire voxel size; x,y in 4 nm, z in 40 nm sections.
VOXEL_NM = np.array([4.0, 4.0, 40.0])


def load_tables():
    comp = pd.read_csv(COMP, index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    idx_of = pd.Series(np.arange(len(root_ids)), index=root_ids)

    ann = pd.read_csv(ANN, sep="\t", low_memory=False)
    ann = ann[ann.root_id.isin(idx_of.index)].copy()
    ann["idx"] = idx_of.reindex(ann.root_id).to_numpy()
    ann["cell_type"] = ann["cell_type"].astype("string")
    ann = ann.drop_duplicates(subset="idx").set_index("idx", drop=False)

    t = pq.read_table(
        CONN,
        columns=[
            "Presynaptic_Index",
            "Postsynaptic_Index",
            "Excitatory x Connectivity",
            "Connectivity",
        ],
    )
    edges = dict(
        pre=t.column(0).to_numpy(),
        post=t.column(1).to_numpy(),
        w=t.column(2).to_numpy().astype(np.float64),
        n=t.column(3).to_numpy().astype(np.float64),
    )
    return root_ids, ann, edges


def stratified_pick(sub: pd.DataFrame, cap: int, axis_col: str) -> np.ndarray:
    """Keep `cap` neurons spanning the population's anatomical axis.

    Populations like LC4 are retinotopic sheets; picking evenly along the
    anterior-posterior axis preserves visual-field coverage rather than
    clustering the survivors in one corner of the map.
    """
    if len(sub) <= cap:
        return sub.idx.to_numpy()
    order = sub.sort_values(axis_col, kind="stable")
    take = np.linspace(0, len(order) - 1, cap).round().astype(int)
    return order.idx.to_numpy()[np.unique(take)]


def select_neurons(ann: pd.DataFrame):
    """Return (records, keep_factor) for every neuron kept by the spec."""
    records: dict[int, dict] = {}
    factor: dict[int, float] = {}

    def add(idx_array, role, n_total, cell_type):
        scale = n_total / max(len(idx_array), 1)
        for i in idx_array:
            i = int(i)
            if i in records:
                continue
            row = ann.loc[i]
            records[i] = {
                "idx": i,
                "root_id": int(row.root_id),
                "cell_type": str(row.cell_type),
                "group": cell_type,
                "side": str(row.side),
                "role": role,
                "nt": str(row.top_nt),
                "pos": [float(row.pos_x), float(row.pos_y), float(row.pos_z)],
            }
            factor[i] = scale

    def by_type(cell_type, role, cap, prefix=False):
        if prefix:
            sel = ann[ann.cell_type.str.startswith(cell_type, na=False)]
        else:
            sel = ann[ann.cell_type == cell_type]
        if sel.empty:
            print(f"  ! no neurons for {cell_type!r}")
            return
        for side, sub in sel.groupby("side"):
            if cap is None:
                kept = sub.idx.to_numpy()
            else:
                kept = stratified_pick(sub, cap, "pos_z")
            add(kept, role, len(sub), cell_type)

    print("selecting visual projection neurons...")
    for ct, role, cap in spec.VISUAL:
        by_type(ct, role, cap)

    print("selecting landmark pathway (MeTu -> TuBu -> ER -> EPG)...")
    for ct, role, cap in spec.LANDMARK:
        by_type(ct, role, cap)
    for pfx, role in spec.LANDMARK_COMPLETE_PREFIX:
        for ct in sorted(set(ann.cell_type.dropna()[ann.cell_type.str.startswith(pfx, na=False)])):
            by_type(ct, role, None)

    print("selecting central complex...")
    for ct, role in spec.CX_COMPLETE:
        by_type(ct, role, None)
    for pfx, role in spec.CX_COMPLETE_PREFIX:
        for ct in sorted(set(ann.cell_type.dropna()[ann.cell_type.str.startswith(pfx, na=False)])):
            by_type(ct, role, None)
    for ct, role, cap in spec.CX_CAPPED:
        by_type(ct, role, cap)
    for pfx, role, cap in spec.CX_PREFIX_CAPPED:
        for ct in sorted(set(ann.cell_type.dropna()[ann.cell_type.str.startswith(pfx, na=False)])):
            by_type(ct, role, cap)

    print("selecting named descending neurons...")
    for ct in spec.DN_ALWAYS:
        by_type(ct, spec.ROLE_DN, None)

    return records, factor


def add_connected_dns(ann, edges, records, factor, min_syn=120, cap=140):
    """Keep additional descending neurons strongly driven by the core."""
    core = np.fromiter(records.keys(), dtype=np.int64)
    is_core = np.zeros(ann.idx.max() + 1, dtype=bool)
    is_core[core] = True
    dn_idx = ann[ann.super_class == "descending"].idx.to_numpy()
    is_dn = np.zeros_like(is_core)
    is_dn[dn_idx] = True

    pre, post, n = edges["pre"], edges["post"], edges["n"]
    m = is_core[pre] & is_dn[post] & ~is_core[post]
    drive = np.zeros(len(is_core))
    np.add.at(drive, post[m], n[m])
    cand = np.argsort(-drive)[:cap]
    cand = [int(i) for i in cand if drive[i] >= min_syn]
    print(f"  + {len(cand)} connectivity-selected DNs (>= {min_syn} synapses from core)")
    for i in cand:
        row = ann.loc[i]
        records[i] = {
            "idx": i,
            "root_id": int(row.root_id),
            "cell_type": str(row.cell_type),
            "group": str(row.cell_type),
            "side": str(row.side),
            "role": spec.ROLE_DN,
            "nt": str(row.top_nt),
            "pos": [float(row.pos_x), float(row.pos_y), float(row.pos_z)],
        }
        factor[i] = 1.0


def add_recruits(ann, records, factor, path, limit):
    """Add the neurons the whole-brain model activates but the spec missed.

    Produced by pipeline/find_recruits.py.  These are not guesses: each one
    fired above threshold in the full 138 639-neuron simulation driven by the
    same stimuli the mod uses, so leaving them out is what made the truncated
    circuit run hot (see docs/METHODS.md).
    """
    if limit <= 0 or not Path(path).exists():
        return
    with open(path) as f:
        rids = json.load(f)["root_ids"][:limit]
    by_root = ann.set_index("root_id")
    added = 0
    for rid in rids:
        if rid not in by_root.index:
            continue
        row = by_root.loc[rid]
        i = int(row.idx)
        if i in records:
            continue
        # a recruited descending neuron is still a motor output, so it must
        # join the read-out population rather than sit in a generic bucket
        role = spec.ROLE_DN if row.super_class == "descending" else "recruited"
        records[i] = {
            "idx": i,
            "root_id": int(rid),
            "cell_type": str(row.cell_type),
            "group": str(row.cell_type),
            "side": str(row.side),
            "role": role,
            "nt": str(row.top_nt),
            "pos": [float(row.pos_x), float(row.pos_y), float(row.pos_z)],
        }
        factor[i] = 1.0
        added += 1
    print(f"  + {added} recruited neurons (active in the whole-brain model)")


def add_bridges(ann, edges, records, factor, budget):
    """Keep interneurons that carry strong real pathways between core groups."""
    size = int(ann.idx.max()) + 1
    is_core = np.zeros(size, dtype=bool)
    is_core[np.fromiter(records.keys(), dtype=np.int64)] = True
    pre, post, n = edges["pre"], edges["post"], edges["n"]

    inflow = np.zeros(size)
    outflow = np.zeros(size)
    m_in = is_core[pre] & (post < size)
    np.add.at(inflow, post[m_in], n[m_in])
    m_out = is_core[post] & (pre < size)
    np.add.at(outflow, pre[m_out], n[m_out])

    score = np.minimum(inflow, outflow)
    score[is_core] = 0
    ok = score >= spec.BRIDGE_MIN_SYN
    cand = np.argsort(-score)[:budget]
    cand = [int(i) for i in cand if ok[i] and i in ann.index]
    print(f"  + {len(cand)} bridge interneurons (>= {spec.BRIDGE_MIN_SYN} syn both ways)")
    for i in cand:
        row = ann.loc[i]
        records[i] = {
            "idx": i,
            "root_id": int(row.root_id),
            "cell_type": str(row.cell_type),
            "group": str(row.cell_type),
            "side": str(row.side),
            "role": spec.ROLE_BRIDGE,
            "nt": str(row.top_nt),
            "pos": [float(row.pos_x), float(row.pos_y), float(row.pos_z)],
        }
        factor[i] = 1.0


def visual_map(records):
    """Assign azimuth/elevation to visual neurons from their lobula position.

    Axes were identified empirically from the annotation table using landmarks
    (see docs/METHODS.md): x = left->right, y = dorsal->ventral,
    z = anterior->posterior.  Within one optic lobe the lobula is retinotopic,
    so z maps to visual azimuth (anterior = frontal field) and y to elevation.
    Coordinates are normalised per (cell_type, side) to [-1, 1].
    """
    visual_roles = {spec.ROLE_LOOMING, spec.ROLE_SMALL_OBJ, spec.ROLE_BAR,
                    spec.ROLE_FLOW, spec.ROLE_LANDMARK}
    by_group: dict[tuple, list] = {}
    for r in records.values():
        if r["role"] in visual_roles:
            by_group.setdefault((r["group"], r["side"]), []).append(r)

    for (group, side), rs in by_group.items():
        pos = np.array([r["pos"] for r in rs]) * VOXEL_NM
        for key, col in (("az", 2), ("el", 1)):
            v = pos[:, col]
            lo, hi = v.min(), v.max()
            norm = np.zeros(len(v)) if hi - lo < 1e-6 else 2 * (v - lo) / (hi - lo) - 1
            for r, x in zip(rs, norm):
                r[key] = float(x)
        # Elevation: y grows ventrally, so flip to make +1 = dorsal visual field.
        for r in rs:
            r["el"] = -r["el"]
            # Azimuth in head-centred degrees.  Each eye of Drosophila covers
            # roughly 170 deg, from about 10 deg across the midline (so the two
            # eyes overlap binocularly in front) round to 160 deg behind.
            hemi = 1.0 if side == "right" else -1.0
            r["az_deg"] = float(hemi * (-10.0 + 85.0 * (r["az"] + 1.0)))


def build_csr(records, factor, edges, w_syn_mV=0.275):
    """Signed weight matrix over kept neurons, in mV per presynaptic spike."""
    kept = np.array(sorted(records.keys()), dtype=np.int64)
    local_of = -np.ones(int(edges["pre"].max()) + 2, dtype=np.int64)
    local_of[kept] = np.arange(len(kept))

    pre, post, w, n = edges["pre"], edges["post"], edges["w"], edges["n"]
    m = (local_of[pre] >= 0) & (local_of[post] >= 0)
    lp, lq = local_of[pre[m]], local_of[post[m]]
    scale = np.array([factor[int(i)] for i in kept])
    vals = w[m] * scale[lp] * w_syn_mV

    mat = sp.csr_matrix((vals, (lp, lq)), shape=(len(kept), len(kept)))
    mat.sum_duplicates()

    # Diagnostic: how much of each kept neuron's real input survived truncation.
    total_in = np.zeros(int(post.max()) + 2)
    np.add.at(total_in, post, n)
    kept_in = np.zeros(len(kept))
    np.add.at(kept_in, lq, n[m])
    cover = kept_in / np.maximum(total_in[kept], 1)
    return kept, mat, cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build/circuit.json"))
    ap.add_argument("--recruits", default=str(ROOT / "build/recruits.json"))
    ap.add_argument("--max-recruits", type=int, default=0,
                    help="neurons to re-add from find_recruits.py. Measured to "
                         "HURT fidelity: each added shell arrives without its "
                         "own inhibitory inputs, so it excites the circuit "
                         "without braking it (see docs/METHODS.md). Default 0.")
    args = ap.parse_args()

    print("loading FlyWire v783 tables...")
    root_ids, ann, edges = load_tables()
    print(f"  {len(root_ids)} neurons, {len(edges['pre'])} edges, {len(ann)} annotated")

    records, factor = select_neurons(ann)
    print(f"  core after spec: {len(records)} neurons")
    add_connected_dns(ann, edges, records, factor)
    add_recruits(ann, records, factor, args.recruits, args.max_recruits)
    per_round = spec.MAX_BRIDGE // spec.BRIDGE_ROUNDS
    for rnd in range(spec.BRIDGE_ROUNDS):
        print(f"  bridge round {rnd + 1}:")
        add_bridges(ann, edges, records, factor, per_round)
    print(f"  total selected: {len(records)} neurons")

    visual_map(records)
    kept, mat, cover = build_csr(records, factor, edges)

    local = {int(g): i for i, g in enumerate(kept)}
    neurons = []
    for g in kept:
        r = records[int(g)]
        neurons.append(
            {
                "i": local[int(g)],
                "root_id": r["root_id"],
                "type": r["cell_type"],
                "group": r["group"],
                "side": r["side"],
                "role": r["role"],
                "nt": r["nt"],
                "az": round(r.get("az", 0.0), 4),
                "el": round(r.get("el", 0.0), 4),
                "az_deg": round(r.get("az_deg", 0.0), 2),
                "scale": round(factor[int(g)], 4),
                "in_cover": round(float(cover[local[int(g)]]), 4),
            }
        )

    csr = mat.tocsr()
    out = {
        "meta": {
            "source": "FlyWire v783 (Dorkenwald et al. 2024) + annotations (Schlegel et al. 2024)",
            "model": "Shiu et al. 2024 LIF, w_syn = 0.275 mV per synapse",
            "n_neurons": int(len(kept)),
            "n_edges": int(csr.nnz),
            "n_synapses_equiv": float(abs(mat).sum() / 0.275),
        },
        "neurons": neurons,
        "csr": {
            "indptr": csr.indptr.tolist(),
            "indices": csr.indices.tolist(),
            "data": [round(float(x), 5) for x in csr.data],
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)

    print(f"\nwrote {args.out}")
    print(f"  neurons  : {len(kept)}")
    print(f"  edges    : {csr.nnz}")
    print(f"  in-cover : median {np.median(cover):.2f}  (fraction of real input retained)")
    roles = pd.Series([n["role"] for n in neurons]).value_counts()
    print("  roles    :", roles.to_dict())


if __name__ == "__main__":
    main()
