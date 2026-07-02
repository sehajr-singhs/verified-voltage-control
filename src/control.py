"""Reactive-power droop controllers for the LinDistFlow feeder.

All controllers are decentralized: each bus updates its own reactive injection q_i from its own
local voltage measurement v_i only. Every controller shares the droop update (Eq. 3 / Eq. 8)

    q_{t+1} = q_t - K (v_t - v_t^ref),

with a diagonal gain K, a per-step rate limit (the reactive-power saturation of the paper), an
absolute reactive-capacity cap, and a small deadband. They differ only in how v_t^ref is set:

  * FixedDroop            v^ref = 1 (the baseline that re-fights every mode transition).
  * SwitchingRefOracle    v^ref = b +/- (1/2) Delta_v_mode from Prop. 2/3, using the known model.
  * SwitchingRefAdaptive  v^ref = v^bias + s v^amp with v^amp, v^bias, s estimated from the local
                          voltage window only (Section III-C, Eqs. 15-17). This is deployable.
"""

from __future__ import annotations

import numpy as np


def _droop_update(q, v, ref, kvec, dq_max, q_cap, deadband):
    """One decentralized droop step with deadband, rate limit, and capacity cap."""
    e = v - ref
    e = np.where(np.abs(e) < deadband, 0.0, e)   # deadband
    dq = -kvec * e
    dq = np.clip(dq, -dq_max, dq_max)            # reactive-power saturation (rate limit)
    return np.clip(q + dq, -q_cap, q_cap)        # capacity cap


class FixedDroop:
    """Baseline: fixed reference at 1 p.u. (paper Eq. 3)."""

    name = "fixed"

    def __init__(self, n, kvec, dq_max=0.01, q_cap=0.30, deadband=0.02):
        self.n = n
        self.kvec = np.broadcast_to(kvec, (n,)).copy()
        self.dq_max, self.q_cap, self.deadband = dq_max, q_cap, deadband
        self.reset()

    def reset(self):
        self.q = np.zeros(self.n)

    def reference(self, v, t):
        return np.ones(self.n)

    def step(self, v, t):
        ref = self.reference(v, t)
        self.q = _droop_update(self.q, v, ref, self.kvec, self.dq_max, self.q_cap, self.deadband)
        return self.q, ref


class SwitchingRefOracle(FixedDroop):
    """Switching reference from Prop. 2/3 using the known mode schedule and feeder model.

    Delta_v_mode = R (pbar(0) - pbar(1)) cancels the mode-transition disturbance (Prop. 2), and
    the bias b centers the two mode envelopes on 1 p.u. (Prop. 3). The mode label comes from the
    workload schedule, so this is the model-based reference used as ground truth in E3, not the
    deployable local-only controller.
    """

    name = "switching-oracle"

    def __init__(self, feeder, workload, kvec, **kw):
        super().__init__(len(feeder.buses), kvec, **kw)
        self.workload = workload
        gap = workload.mode_injection_gap(feeder)          # pbar_inj(0) - pbar_inj(1)
        self.delta_v_mode = feeder.R @ gap                 # Prop. 2
        self.bias = self._prop3_bias(feeder, workload)     # Prop. 3

    def _prop3_bias(self, feeder, workload):
        # Prop. 3 updates the bias by b <- b - ((v+ + v-)/2 - 1) until the two mode voltages
        # straddle 1 p.u. Because the integral droop drives each mode's steady-state voltage to
        # its reference b +/- (1/2) Delta_v_mode, the two voltages average to b, so the fixed
        # point of the update is exactly b* = 1. We use that fixed point directly.
        return np.ones(len(feeder.buses))

    def reference(self, v, t):
        m = self.workload.mode(t)
        # mode 0 (comm, higher voltage) -> higher reference; mode 1 (compute) -> lower.
        return self.bias + (0.5 * self.delta_v_mode if m == 0 else -0.5 * self.delta_v_mode)


class SwitchingRefAdaptive(FixedDroop):
    """Deployable switching reference estimated from the local voltage window (Section III-C).

    Amplitude (Eq. 15) is half the largest recent single-step voltage jump, which is set by the
    mode transition and is not amplified by the controller because within the one step of a
    transition the reactive dispatch has not yet moved. Bias (Eq. 16) drifts the reference center
    toward 1 p.u. from the window max/min. Sign (Eq. 17) selects the mode from the local voltage.

    Two guards keep the local estimator stable, both documented as deviations from the bare paper
    equations. The sign uses hysteresis (a margin band around the bias) so it does not chatter when
    the voltage sits near the bias, which would otherwise flip the reference by 2*amp every step and
    drive the droop unstable. The amplitude and bias are clipped to physical ranges (amp to the
    +/-5% band, bias to within 5% of nominal) so a startup transient cannot blow the estimate up.
    """

    name = "switching-adaptive"

    def __init__(self, n, kvec, La=120, Lb=120, eta_b=0.02, amp_cap=0.05, **kw):
        super().__init__(n, kvec, **kw)
        self.La, self.Lb, self.eta_b, self.amp_cap = La, Lb, eta_b, amp_cap

    def reset(self):
        super().reset()
        self.bias = np.ones(self.n)
        self.amp = np.zeros(self.n)
        self.s = np.ones(self.n)                           # held mode sign (hysteretic)
        self.hist = []                                     # recent voltage vectors

    def _update_estimates(self):
        H = np.array(self.hist)                            # (window, n)
        if len(H) >= 2:
            diffs = np.abs(np.diff(H[-self.La:], axis=0))  # |v_{t-l} - v_{t-l-1}|  (Eq. 15)
            self.amp = np.clip(0.5 * diffs.max(axis=0), 0.0, self.amp_cap)
        W = H[-self.Lb:]                                    # bias window (Eq. 16)
        db = 0.5 * (W.max(axis=0) + W.min(axis=0)) - 1.0
        self.bias = np.clip(self.bias - self.eta_b * db, 0.95, 1.05)

    def reference(self, v, t):
        margin = 0.3 * self.amp                            # hysteresis on the sign selection
        self.s = np.where(v > self.bias + margin, 1.0,
                          np.where(v < self.bias - margin, -1.0, self.s))  # Eq. 17 with hysteresis
        return self.bias + self.s * self.amp

    def step(self, v, t):
        self.hist.append(np.asarray(v, dtype=float).copy())
        if len(self.hist) > max(self.La, self.Lb) + 2:
            self.hist.pop(0)
        self._update_estimates()
        ref = self.reference(v, t)
        self.q = _droop_update(self.q, v, ref, self.kvec, self.dq_max, self.q_cap, self.deadband)
        return self.q, ref


def _dc_inj(feeder, workload, mode):
    p = np.zeros(len(feeder.buses))
    for bus in workload.dc_buses:
        p[feeder.idx[bus]] = -workload.pbar(mode)
    return p


def max_stable_gain(X):
    """Largest scalar k with 0 < kI < 2 X^{-1}, i.e. k < 2 / lambda_max(X) (Theorem 1)."""
    return 2.0 / np.linalg.eigvalsh(X).max()
