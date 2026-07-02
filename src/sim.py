"""Closed-loop simulation of the feeder under a droop controller and a data-center workload."""

from __future__ import annotations

import numpy as np

from src.load import nominal_injections

VOLT_BAND = 0.05   # +/- 5% per-unit voltage band


def simulate(feeder, controller, workload, T, warmup=0):
    """Run the discrete closed loop for T control steps.

    At each step the feeder voltage is the LinDistFlow map of the total injection given the
    control's current reactive dispatch, the controller measures that voltage, and then updates
    its dispatch for the next step. Summary metrics are computed on the post-warmup window (steady
    operation) so the initial dispatch-ramp transient does not dominate; full trajectories, warmup
    included, are always returned.
    """
    p_base, q_base = nominal_injections(feeder)
    controller.reset()
    n = len(feeder.buses)

    V = np.zeros((T, n))
    Q = np.zeros((T, n))
    REF = np.zeros((T, n))
    MODE = np.zeros(T, dtype=int)

    q_prev = controller.q.copy()
    dq_total = 0.0
    for t in range(T):
        p = p_base + workload.load_at(feeder, t)
        v = feeder.voltage(p, q_base + controller.q)
        V[t] = v
        q_new, ref = controller.step(v, t)
        Q[t] = q_new
        REF[t] = ref
        MODE[t] = workload.mode(t)
        dq_total += np.abs(q_new - q_prev).sum()
        q_prev = q_new

    Vw = V[warmup:]
    dev = np.abs(Vw - 1.0)
    metrics = {
        "max_dev": float(dev.max()),
        "violation_frac": float((dev > VOLT_BAND).mean()),
        "n_violations": int((dev > VOLT_BAND).sum()),
        "dq_total": float(np.abs(np.diff(Q[warmup:], axis=0)).sum()),
        "mean_abs_q": float(np.abs(Q[warmup:]).mean()),
        "v_min": float(Vw.min()),
        "v_max": float(Vw.max()),
        "startup_max_dev": float(np.abs(V[:warmup] - 1.0).max()) if warmup else 0.0,
        "warmup": int(warmup),
    }
    traj = {"V": V, "Q": Q, "REF": REF, "MODE": MODE}
    return metrics, traj
