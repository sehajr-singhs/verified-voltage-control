"""Load model: nominal 33-bus demand plus a two-mode AI-training data-center workload.

The paper's data center switches synchronously between a high-power compute phase and a
low-power communication phase, producing periodic step changes in active power (Eq. 2):

    p_{j,t} = pbar_j(m_t) + w_t,   m_t in {0,1},   |w_t| <= wbar(m_t).

We drive the feeder with two components. First, the standard Baran-Wu nominal bus loads,
held constant, which set a realistic base voltage profile. Second, a data-center active-power
load at bus 22 (and bus 25 in the two-DC case) that switches between the two modes with bounded
intra-mode fluctuation. The real paper uses a measured ~45-minute DGX/H200 QLoRA trace; we do
not have that trace, so we model the workload as a periodic square wave between the two mode
levels with uniform intra-mode noise, and we flag this as a modeled (not measured) trace in the
limitations. The switching structure, not the exact waveform, is what the controller and the
certificate depend on.
"""

from __future__ import annotations

import numpy as np

# Standard Baran-Wu 33-bus nominal loads: bus -> (P_kW, Q_kVar). Bus 1 (slack) has none.
NOMINAL_LOAD_KW_KVAR = {
    2: (100, 60), 3: (90, 40), 4: (120, 80), 5: (60, 30), 6: (60, 20), 7: (200, 100),
    8: (200, 100), 9: (60, 20), 10: (60, 20), 11: (45, 30), 12: (60, 35), 13: (60, 35),
    14: (120, 80), 15: (60, 10), 16: (60, 20), 17: (60, 20), 18: (90, 40), 19: (90, 40),
    20: (90, 40), 21: (90, 40), 22: (90, 40), 23: (90, 50), 24: (420, 200), 25: (420, 200),
    26: (60, 25), 27: (60, 25), 28: (60, 20), 29: (120, 70), 30: (200, 600), 31: (150, 70),
    32: (210, 100), 33: (60, 40),
}


# The textbook 33-bus is a stressed feeder (min ~0.913 p.u. before any compensation), so a few
# far laterals sit below -5% even at base load, which would mask the data-center effect we study.
# We scale the nominal loads by LOAD_SCALE so the uncompensated base profile sits inside the band,
# making the data-center switching the clear cause of the band excursions. This is a modeling
# choice, stated plainly in the limitations; the topology, R/X, and control math are unchanged.
LOAD_SCALE = 0.5


def nominal_injections(feeder, scale=LOAD_SCALE):
    """Constant background injections p_base, q_base (p.u.), loads as negative injections."""
    p = np.zeros(len(feeder.buses))
    q = np.zeros(len(feeder.buses))
    base_kw = feeder_base_kw(feeder)
    for bus, (pk, qk) in NOMINAL_LOAD_KW_KVAR.items():
        i = feeder.idx[bus]
        p[i] = -scale * pk / base_kw
        q[i] = -scale * qk / base_kw
    return p, q


def feeder_base_kw(feeder):
    from src.grid import BASE_MVA
    return BASE_MVA * 1000.0  # 10 MVA -> 10000 kW


class DCWorkload:
    """Two-mode data-center active-power load with periodic switching and bounded noise.

    Parameters are in per unit on the 10 MVA base. compute_pu is the compute-phase (mode 1)
    active load magnitude, comm_pu the communication-phase (mode 0) magnitude, both positive
    numbers representing a load (subtracted from injection). The load steps between them with
    the given phase durations; wbar is the half-width of the uniform intra-mode fluctuation.
    """

    def __init__(self, dc_buses, compute_pu=0.20, comm_pu=0.04,
                 compute_steps=40, comm_steps=20, wbar=0.010, seed=0):
        self.dc_buses = list(dc_buses)
        self.compute_pu = compute_pu
        self.comm_pu = comm_pu
        self.compute_steps = compute_steps
        self.comm_steps = comm_steps
        self.wbar = wbar
        self.rng = np.random.default_rng(seed)
        self.period = compute_steps + comm_steps

    def mode(self, t):
        """m_t in {0,1}: 1 during the compute phase, 0 during communication."""
        return 1 if (t % self.period) < self.compute_steps else 0

    def pbar(self, m):
        """Mean data-center load magnitude (p.u.) for mode m."""
        return self.compute_pu if m == 1 else self.comm_pu

    def load_at(self, feeder, t):
        """Return the DC contribution to the injection vector p at step t (negative = load)."""
        p = np.zeros(len(feeder.buses))
        m = self.mode(t)
        for bus in self.dc_buses:
            w = self.rng.uniform(-self.wbar, self.wbar)
            p[feeder.idx[bus]] = -(self.pbar(m) + w)
        return p

    def mode_injection_gap(self, feeder):
        """pbar(0) - pbar(1) as an injection vector at the DC buses (used by Prop. 2)."""
        gap = np.zeros(len(feeder.buses))
        for bus in self.dc_buses:
            # injection = -load, so pbar_inj(0) - pbar_inj(1) = -(comm) - (-(compute)) = compute - comm
            gap[feeder.idx[bus]] = -(self.comm_pu) - (-(self.compute_pu))
        return gap
