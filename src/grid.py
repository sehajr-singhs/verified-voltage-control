"""LinDistFlow model of the IEEE 33-bus radial distribution feeder.

The paper (Yan, Joswig-Jones, Zhang, Chen, Cui, arXiv:2603.15588, Eq. 1) writes the
linearized voltage as

    v_t = R p_t + X q_t + 1,

with R, X symmetric positive definite and 1 the substation reference at 1 p.u. This
module builds R and X from the network topology so that everything downstream (control,
certification) runs on the real feeder matrices rather than a toy.

Derivation. On a radial feeder LinDistFlow (Baran and Wu, 1989) linearizes the DistFlow
equations by dropping the quadratic loss terms. The squared voltage at bus i then satisfies
    u_i = u_0 - 2 * sum_{l in path(i)} ( r_l P_l + x_l Q_l ),
where P_l, Q_l are the active/reactive powers flowing on branch l and path(i) is the set of
branches on the unique root-to-i path. Substituting P_l = sum of downstream injections gives
the closed form
    u = 1 + R p + X q,   R_ij = 2 * sum_{l in path(i) ∩ path(j)} r_l,
                          X_ij = 2 * sum_{l in path(i) ∩ path(j)} x_l,
with p, q the active/reactive power *injections* (loads enter as negative p). Near 1 p.u.
u = v^2 ≈ 1 + 2(v-1), so we work directly with v in per unit and keep the factor 2 in R, X,
which makes them the standard LinDistFlow sensitivity matrices. Both are symmetric positive
definite because the path-overlap (common-ancestor) structure is a Gram matrix of a tree.

Branch data is the standard Baran-Wu 33-bus system, base 10 MVA / 12.66 kV, R and X in ohms.
Source: M. E. Baran and F. F. Wu, "Network reconfiguration in distribution systems for loss
reduction and load balancing," IEEE Trans. Power Delivery, 4(2), 1989. The same feeder is used
by Prof. Cui's decentralized safe-RL voltage-control code (github.com/Wenqi-Cui/Voltage-Control).
"""

from __future__ import annotations

import numpy as np

# Baran-Wu IEEE 33-bus branch data: (from_bus, to_bus, R_ohm, X_ohm). Buses 1..33, bus 1 slack.
# 32 branches, radial. "from" is the parent (closer to the substation), "to" is the child.
BRANCHES = [
    (1, 2, 0.0922, 0.0470), (2, 3, 0.4930, 0.2511), (3, 4, 0.3660, 0.1864),
    (4, 5, 0.3811, 0.1941), (5, 6, 0.8190, 0.7070), (6, 7, 0.1872, 0.6188),
    (7, 8, 1.7114, 1.2351), (8, 9, 1.0300, 0.7400), (9, 10, 1.0400, 0.7400),
    (10, 11, 0.1966, 0.0650), (11, 12, 0.3744, 0.1238), (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129), (14, 15, 0.5910, 0.5260), (15, 16, 0.7463, 0.5450),
    (16, 17, 1.2890, 1.7210), (17, 18, 0.7320, 0.5740), (2, 19, 0.1640, 0.1565),
    (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784), (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083), (23, 24, 0.8980, 0.7091), (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.0590, 0.9337),
    (28, 29, 0.8042, 0.7006), (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.9630),
    (31, 32, 0.3105, 0.3619), (32, 33, 0.3410, 0.5302),
]

BASE_MVA = 10.0
BASE_KV = 12.66
N_BUS = 33            # including the slack bus 1
N = N_BUS - 1         # number of controllable/non-slack buses (2..33)


def build_matrices(branches, z_base, root=1, factor=1.0):
    """Build LinDistFlow R, X from radial branch data.

    branches: iterable of (from_bus, to_bus, R_ohm, X_ohm), from_bus the parent.
    Returns (buses, idx, R, X) with buses the sorted non-root bus list and idx a
    bus-number -> matrix-index map. R_ij = factor * sum of branch resistances shared by the
    root->i and root->j paths (likewise X), which is the path-overlap Gram matrix.

    factor=1 gives the voltage-magnitude sensitivity v_i ≈ 1 - sum_j(R_ij P_load + X_ij Q_load),
    which reproduces the textbook 33-bus profile (min ~0.913 p.u. at bus 18). The paper states
    the +/-5% band on the voltage magnitude v, so we use factor=1. (factor=2 would give the
    squared-voltage form u = v^2; near 1 p.u. the two differ only by the well-known factor 2.)
    """
    parent = {t: (f, r / z_base, x / z_base) for f, t, r, x in branches}
    buses = sorted(parent.keys())
    idx = {b: i for i, b in enumerate(buses)}

    def path(bus):
        edges = set()
        b = bus
        while b != root:
            edges.add(b)          # edge (parent(b), b) keyed by its child b
            b = parent[b][0]
        return edges

    path_edges = {b: path(b) for b in buses}
    r = {b: parent[b][1] for b in buses}
    x = {b: parent[b][2] for b in buses}
    n = len(buses)
    R = np.zeros((n, n))
    X = np.zeros((n, n))
    for bi in buses:
        for bj in buses:
            common = path_edges[bi] & path_edges[bj]
            R[idx[bi], idx[bj]] = factor * sum(r[e] for e in common)
            X[idx[bi], idx[bj]] = factor * sum(x[e] for e in common)
    return buses, idx, R, X


class Feeder:
    """IEEE 33-bus LinDistFlow feeder. Indexes non-slack buses 2..33 as 0..N-1."""

    def __init__(self, branches=BRANCHES):
        z_base = (BASE_KV ** 2) / BASE_MVA  # ohms
        self.z_base = z_base
        self.buses, self.idx, self.R, self.X = build_matrices(branches, z_base)
        self._check_spd()

    def _check_spd(self):
        for name, M in (("R", self.R), ("X", self.X)):
            assert np.allclose(M, M.T), f"{name} not symmetric"
            w = np.linalg.eigvalsh(M)
            assert w.min() > 0, f"{name} not positive definite (min eig {w.min():.3e})"

    def voltage(self, p, q):
        """LinDistFlow per-unit voltage v = R p + X q + 1 (Eq. 1). p, q are injections."""
        return self.R @ np.asarray(p) + self.X @ np.asarray(q) + 1.0

    def bus_index(self, bus_number):
        return self.idx[bus_number]


if __name__ == "__main__":
    f = Feeder()
    print(f"IEEE 33-bus feeder: {N} non-slack buses, z_base = {(BASE_KV**2)/BASE_MVA:.3f} ohm")
    print(f"R spd: eig in [{np.linalg.eigvalsh(f.R).min():.4e}, {np.linalg.eigvalsh(f.R).max():.4e}]")
    print(f"X spd: eig in [{np.linalg.eigvalsh(f.X).min():.4e}, {np.linalg.eigvalsh(f.X).max():.4e}]")
    # A 1 MW (0.1 p.u.) load at bus 22, no reactive support, should pull bus 22 down the most.
    p = np.zeros(N); p[f.bus_index(22)] = -0.1
    v = f.voltage(p, np.zeros(N))
    print(f"1 MW load at bus 22 -> min voltage {v.min():.4f} p.u. at bus {f.buses[int(np.argmin(v))]}")
