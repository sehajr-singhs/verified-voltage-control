"""Tests for the droop controllers and the Theorem 1 gain condition."""

import numpy as np

from src.grid import Feeder
from src.control import (FixedDroop, SwitchingRefOracle, SwitchingRefAdaptive,
                         max_stable_gain)
from src.load import DCWorkload


def test_gain_satisfies_theorem1_condition():
    f = Feeder()
    kmax = max_stable_gain(f.X)
    for frac in (0.5, 0.7, 0.9):
        k = frac * kmax
        # Theorem 1: 0 < kI < 2 X^-1  <=>  A = I - k X has spectral radius < 1.
        A = np.eye(f.X.shape[0]) - k * f.X
        assert np.abs(np.linalg.eigvals(A)).max() < 1.0
    # At exactly kmax the fastest mode sits on the unit circle.
    A = np.eye(f.X.shape[0]) - kmax * f.X
    assert np.isclose(np.abs(np.linalg.eigvals(A)).max(), 1.0, atol=1e-6)


def test_rate_and_capacity_limits_respected():
    n = 5
    c = FixedDroop(n, 100.0, dq_max=0.01, q_cap=0.05, deadband=0.0)
    v = np.full(n, 0.5)          # huge error to force saturation
    q_prev = c.q.copy()
    for _ in range(20):
        q, _ = c.step(v, 0)
        assert (np.abs(q - q_prev) <= 0.01 + 1e-12).all()   # rate limit
        assert (np.abs(q) <= 0.05 + 1e-12).all()            # capacity cap
        q_prev = q


def test_deadband_blocks_small_errors():
    n = 3
    c = FixedDroop(n, 1.0, dq_max=1.0, q_cap=1.0, deadband=0.02)
    v = np.full(n, 1.01)         # 0.01 error, inside the 0.02 deadband
    q, _ = c.step(v, 0)
    assert np.allclose(q, 0.0)


def test_oracle_reference_centered_on_nominal():
    f = Feeder()
    wl = DCWorkload([22], compute_pu=0.4, comm_pu=0.03)
    c = SwitchingRefOracle(f, wl, 0.1)
    ref0 = c.reference(np.ones(len(f.buses)), t=0)                 # compute phase (mode 1)
    ref_comm = c.reference(np.ones(len(f.buses)), t=wl.compute_steps)  # comm phase (mode 0)
    j = f.idx[22]
    # The two mode references straddle 1 p.u. symmetrically (bias = 1, Prop. 3 fixed point).
    assert np.isclose(0.5 * (ref0[j] + ref_comm[j]), 1.0, atol=1e-9)
    assert ref_comm[j] > 1.0 > ref0[j]   # comm (light load) high, compute (heavy) low


def test_adaptive_estimates_stay_bounded():
    # The local estimator must not blow up: amplitude within the band, bias within 5%.
    f = Feeder()
    from src.sim import simulate
    c = SwitchingRefAdaptive(len(f.buses), 0.7 * max_stable_gain(f.X), La=90, Lb=90,
                             deadband=0.01)
    m, _ = simulate(f, c, DCWorkload([22], compute_pu=0.4, comm_pu=0.03), 1200, warmup=120)
    assert (c.amp <= 0.05 + 1e-9).all()
    assert (np.abs(c.bias - 1.0) <= 0.05 + 1e-9).all()
    assert m["v_max"] < 1.15 and m["v_min"] > 0.85   # no runaway
