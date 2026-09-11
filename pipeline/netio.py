"""Load build/circuit.json into numpy/scipy structures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent


class Circuit:
    def __init__(self, path=None):
        path = Path(path or ROOT / "build/circuit.json")
        with open(path) as f:
            d = json.load(f)
        self.meta = d["meta"]
        self.neurons = d["neurons"]
        n = len(self.neurons)
        c = d["csr"]
        self.W = sp.csr_matrix(
            (np.array(c["data"]), np.array(c["indices"]), np.array(c["indptr"])),
            shape=(n, n),
        )
        self.N = n
        self.type = np.array([x["type"] for x in self.neurons])
        self.group = np.array([x["group"] for x in self.neurons])
        self.side = np.array([x["side"] for x in self.neurons])
        self.role = np.array([x["role"] for x in self.neurons])
        self.az = np.array([x["az_deg"] for x in self.neurons])
        self.el = np.array([x["el"] for x in self.neurons])
        self.root_id = np.array([x["root_id"] for x in self.neurons], dtype=np.int64)

    def ids(self, type_=None, side=None, role=None, group=None):
        m = np.ones(self.N, bool)
        if type_ is not None:
            m &= np.isin(self.type, np.atleast_1d(type_))
        if group is not None:
            m &= np.isin(self.group, np.atleast_1d(group))
        if side is not None:
            m &= self.side == side
        if role is not None:
            m &= np.isin(self.role, np.atleast_1d(role))
        return np.flatnonzero(m)

    def vrest(self):
        """The per-neuron resting potential the mod uses, reproduced exactly.

        Loads build/bias.npy and the fitted k, and applies the same clamp as
        export_lua.write_bias, so the numpy reference and the Lua engine run
        the identical model.  Returns None if the bias has not been computed.
        """
        import json
        from lif import V_0, V_TH
        bp = ROOT / "build/bias.npy"
        if not bp.exists():
            return None
        k = 0.75
        fp = ROOT / "build/bias_fit.json"
        if fp.exists():
            k = json.loads(fp.read_text())["k"]
        vr = V_0 + np.load(bp) * k
        return np.minimum(vr, V_TH - 2.0)

    def bias(self):
        """vrest() expressed as an offset from the -52 mV resting potential."""
        from lif import V_0
        vr = self.vrest()
        return None if vr is None else vr - V_0

    def label(self, i):
        n = self.neurons[i]
        return f"{n['type']}[{n['side'][:1]}]"
