"""Unit tests for the LinDistFlow feeder: hand-checked small cases + SPD structure."""

import numpy as np

from src.grid import Feeder, build_matrices


def test_three_bus_line_by_hand():
    # root(1) - 2 - 3 line, resistances/reactances in ohms, z_base = 1 so p.u. == ohms.
    r1, x1, r2, x2 = 0.3, 0.2, 0.5, 0.4
    branches = [(1, 2, r1, x1), (2, 3, r2, x2)]
    buses, idx, R, X = build_matrices(branches, z_base=1.0)
    assert buses == [2, 3]
    # R_ij = shared branch resistance on the root->i and root->j paths (magnitude form).
    R_expected = np.array([[r1, r1], [r1, r1 + r2]])
    X_expected = np.array([[x1, x1], [x1, x1 + x2]])
    assert np.allclose(R, R_expected)
    assert np.allclose(X, X_expected)

    # A unit reactive injection at bus 3 lifts bus 3 by (x1+x2) and bus 2 by x1.
    q = np.array([0.0, 1.0])
    v = R @ np.zeros(2) + X @ q + 1.0
    assert np.isclose(v[idx[2]], 1.0 + x1)
    assert np.isclose(v[idx[3]], 1.0 + (x1 + x2))


def test_cross_term_reciprocity():
    # Injection at an upstream bus and a downstream bus share only the common path.
    branches = [(1, 2, 0.1, 0.05), (2, 3, 0.2, 0.1), (3, 4, 0.3, 0.15)]
    _, idx, R, _ = build_matrices(branches, z_base=1.0)
    # R[bus4, bus2] uses only the shared branch (1,2): 0.1.
    assert np.isclose(R[idx[4], idx[2]], 0.1)
    # symmetry
    assert np.allclose(R, R.T)


def test_feeder_spd_and_shape():
    f = Feeder()
    assert f.R.shape == (32, 32)
    assert np.allclose(f.R, f.R.T) and np.allclose(f.X, f.X.T)
    assert np.linalg.eigvalsh(f.R).min() > 0
    assert np.linalg.eigvalsh(f.X).min() > 0


def test_voltage_equation_matches_matrix_form():
    f = Feeder()
    rng = np.random.default_rng(0)
    p = rng.normal(0, 0.05, 32)
    q = rng.normal(0, 0.05, 32)
    assert np.allclose(f.voltage(p, q), f.R @ p + f.X @ q + 1.0)


def test_load_lowers_voltage_monotonically():
    # A pure active load (negative injection) can only lower voltages, most at its own bus.
    f = Feeder()
    p = np.zeros(32)
    p[f.bus_index(18)] = -0.1  # deepest bus on the main trunk
    v = f.voltage(p, np.zeros(32))
    assert (v <= 1.0 + 1e-12).all()
    assert f.buses[int(np.argmin(v))] == 18
