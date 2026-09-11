"""Stimulus protocols shared by the Python reference and the Lua harness.

Each protocol returns a rate (Hz) for every sensory neuron given its
head-centred azimuth and side.  Both implementations must compute the same
numbers, so the arithmetic here is deliberately trivial and duplicated
verbatim in tools/dump_rates.lua.
"""
from __future__ import annotations

import numpy as np

# name -> (role kinds driven, centre azimuth deg, sigma deg, peak Hz, ms)
PROTOCOLS = {
    "loom_right":     (("looming", "small_object"), 45.0, 35.0, 150.0, 1000.0),
    "loom_left":      (("looming", "small_object"), -45.0, 35.0, 150.0, 1000.0),
    "loom_front":     (("looming", "small_object"), 0.0, 30.0, 150.0, 1000.0),
    "landmark_front": (("landmark",), 0.0, 25.0, 120.0, 1000.0),
    "landmark_right": (("landmark",), 60.0, 25.0, 120.0, 1000.0),
    "flow_yaw":       (("optic_flow",), 0.0, 200.0, 80.0, 1000.0),
}


def drive(circuit, name):
    """Return (rates_hz array, no_refractory mask, duration_ms)."""
    kinds, centre, sigma, peak, ms = PROTOCOLS[name]
    hz = np.zeros(circuit.N)
    for i in range(circuit.N):
        if circuit.role[i] not in kinds:
            continue
        d = (circuit.az[i] - centre) / sigma
        r = peak * np.exp(-0.5 * d * d)
        if r > 1.0:
            hz[i] = r
    return hz, hz > 0.0, ms
