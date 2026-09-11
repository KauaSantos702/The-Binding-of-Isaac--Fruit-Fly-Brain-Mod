"""Leaky integrate-and-fire model of Shiu et al. 2024, reimplemented in numpy.

This is the same model `fly-brain/code/paper-phil-drosophila/model.py` builds in
Brian2, written out explicitly so that (a) it runs without Brian2/CUDA, and
(b) the update rule can be transcribed line-for-line into Lua for the mod.

    dv/dt = (v_0 - v + g) / t_mbr      (unless refractory)
    dg/dt = -g / tau                   (unless refractory)
    spike when v > v_th; on spike: v <- v_rst, g <- 0, refractory t_rfc
    on presynaptic spike, after t_dly:  g <- g + w

Integration is exponential-Euler, which is *exact* for this linear system, so
the only approximation versus Brian2 is that spike times are quantised to dt
and the 1.8 ms axonal delay is rounded to a whole number of steps.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Constants from Shiu et al. 2024 / Kakaria & de Bivort 2017, in mV and ms.
V_0 = -52.0      # resting potential
V_RST = -52.0    # reset potential
V_TH = -45.0     # spike threshold
T_MBR = 20.0     # membrane time constant
TAU = 5.0        # synaptic time constant
T_RFC = 2.2      # refractory period
T_DLY = 1.8      # synaptic delay
W_SYN = 0.275    # mV per synapse
F_POI = 250.0    # Poisson synapse scaling
W_POI = W_SYN * F_POI  # 68.75 mV, injected straight into v (as in the paper)


class LIFNetwork:
    """Vectorised LIF network over a sparse signed weight matrix (mV)."""

    def __init__(self, W: sp.csr_matrix, dt: float = 1.0, seed: int = 0,
                 bias: np.ndarray | None = None):
        """`bias` is a per-neuron constant offset to the resting potential, in
        mV, standing in for synaptic input the truncated network lost.  See
        pipeline/compute_bias.py."""
        self.W = W.tocsr()
        self.Wt = self.W.T.tocsr()  # postsyn-major, for gathering incoming input
        self.N = W.shape[0]
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)

        self.ev = np.exp(-self.dt / T_MBR)
        self.eg = np.exp(-self.dt / TAU)
        # analytic solution of dv/dt = (v_0 - v + g)/t_mbr with g decaying at tau
        a = 1.0 / T_MBR - 1.0 / TAU
        self.cvg = (self.eg - self.ev) / (T_MBR * a)

        self.v_rest = np.full(W.shape[0], V_0)
        if bias is not None:
            self.v_rest = self.v_rest + bias

        self.delay_steps = max(1, int(round(T_DLY / self.dt)))
        self.rfc_steps = int(round(T_RFC / self.dt))
        self.reset()

    def reset(self):
        self.v = self.v_rest.copy()
        self.g = np.zeros(self.N)
        self.rfc = np.zeros(self.N, dtype=np.int32)
        self.pending = np.zeros((self.delay_steps + 1, self.N))
        self.slot = 0
        self.t = 0.0

    def step(self, poisson_hz: np.ndarray | None = None, no_refractory: np.ndarray | None = None):
        """Advance one dt.  Returns a boolean array of neurons that spiked."""
        active = self.rfc <= 0
        # state decay (frozen while refractory, as in the Brian2 'unless refractory')
        vr = self.v_rest[active]
        self.v[active] = vr + (self.v[active] - vr) * self.ev + self.g[active] * self.cvg
        self.g[active] *= self.eg

        # deliver synaptic input scheduled for this step
        arrived = self.pending[self.slot]
        self.g += arrived
        arrived.fill(0.0)

        # Poisson drive: as in the paper, each event dumps W_POI straight into v,
        # which is ~10x threshold distance and so reliably evokes a spike.
        if poisson_hz is not None:
            p = poisson_hz * (self.dt / 1000.0)
            n_ev = self.rng.poisson(p)
            hit = n_ev > 0
            if hit.any():
                self.v[hit] += W_POI * n_ev[hit]

        spikes = (self.v > V_TH) & active
        if no_refractory is not None:
            spikes |= (self.v > V_TH) & no_refractory

        if spikes.any():
            idx = np.flatnonzero(spikes)
            self.v[idx] = V_RST
            self.g[idx] = 0.0
            self.rfc[idx] = self.rfc_steps
            # scatter weights of spiking neurons into the delay buffer
            tgt = (self.slot + self.delay_steps) % (self.delay_steps + 1)
            contrib = self.W[idx].sum(axis=0)
            self.pending[tgt] += np.asarray(contrib).ravel()

        if no_refractory is not None:
            self.rfc[no_refractory] = 0
        # decrement only while positive, so the refractory block lasts
        # round(t_rfc/dt) steps; the Lua port does the same.
        np.subtract(self.rfc, 1, out=self.rfc, where=self.rfc > 0)
        self.slot = (self.slot + 1) % (self.delay_steps + 1)
        self.t += self.dt
        return spikes

    def run(self, ms: float, rate_fn=None, no_refractory=None, record=True):
        """Run for `ms` of model time.  `rate_fn(t_ms) -> hz array` sets input."""
        steps = int(round(ms / self.dt))
        rec_t, rec_i = [], []
        counts = np.zeros(self.N, dtype=np.int64)
        for s in range(steps):
            hz = rate_fn(self.t) if rate_fn is not None else None
            spk = self.step(hz, no_refractory)
            counts += spk
            if record and spk.any():
                idx = np.flatnonzero(spk)
                rec_i.append(idx)
                rec_t.append(np.full(len(idx), self.t))
        out = {
            "counts": counts,
            "rates_hz": counts / (ms / 1000.0),
        }
        if record and rec_i:
            out["spike_i"] = np.concatenate(rec_i)
            out["spike_t"] = np.concatenate(rec_t)
        else:
            out["spike_i"] = np.zeros(0, dtype=int)
            out["spike_t"] = np.zeros(0)
        return out
