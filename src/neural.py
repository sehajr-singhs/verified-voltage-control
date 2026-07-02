"""Neural reactive-power controller and its CROWN certification in the LinDistFlow setting.

The neural policy is a small MLP pi_theta that maps the vector of local voltage deviations
e = v - 1 to a reactive-power control injection q = pi_theta(e), trained by behavior cloning the
switching-reference controller. The safety object is the one-step closed-loop deviation

    e^+ = v^+ - 1 = R p + X (q_base + pi_theta(e)) + (X q_base already folded) ... = R p + X pi_theta(e) + c,

a network embedded in the fixed physics (two frozen linear maps R and X around the MLP). Bound
propagation (auto_LiRPA: IBP, backward CROWN, alpha-CROWN) certifies, for EVERY e in an operating
box and EVERY data-center load p in its range, that e^+ stays inside the +/-5% band. This is a
guarantee for all inputs in the set, not a sampled fraction, which is the whole point of the
method and the contrast with a sampled certificate.

We use auto_LiRPA rather than the IBM CROWN-Robustness-Certification repo Prof. Cui pointed to:
it is the same CROWN algorithm family (Zhang et al.), maintained, PyTorch, pip-installable, and
CPU-fine, whereas the 2019 IBM repo is TensorFlow-1-era and does not build on current Python. The
IBM repo is cited as the original in the README.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import gain, make_controller, make_workload
from src.load import nominal_injections
from src.sim import simulate


class Policy(nn.Module):
    """Shared MLP mapping the voltage-deviation vector e (n,) to reactive control q (n,)."""

    def __init__(self, n, hidden=32, q_cap=0.30):
        super().__init__()
        self.q_cap = q_cap
        self.net = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n),
        )

    def forward(self, e):
        return self.net(e)


class OneStep(nn.Module):
    """Composite one-step closed-loop deviation map (e, p) -> e^+ = R p + X pi(e) + c.

    R, X are frozen linear layers (LinDistFlow sensitivities); c = X q_base folds in the constant
    nominal reactive dispatch. This whole module is what CROWN bounds.
    """

    def __init__(self, policy, R, X, q_base):
        super().__init__()
        self.policy = policy
        n = R.shape[0]
        self.Rlin = nn.Linear(n, n, bias=False)
        self.Xlin = nn.Linear(n, n, bias=False)
        self.Rlin.weight = nn.Parameter(torch.tensor(R, dtype=torch.float32), requires_grad=False)
        self.Xlin.weight = nn.Parameter(torch.tensor(X, dtype=torch.float32), requires_grad=False)
        self.register_buffer("c", torch.tensor(X @ q_base, dtype=torch.float32))

    def forward(self, e, p):
        return self.Rlin(p) + self.Xlin(self.policy(e)) + self.c


def collect_clone_data(cfg, feeder, dc_buses, seeds):
    """Collect (e_t, q_t) pairs from the switching-adaptive controller for behavior cloning."""
    E, Q = [], []
    T = cfg["sim"]["T"]
    warmup = cfg["sim"]["warmup"]
    for seed in seeds:
        wl = make_workload(cfg, dc_buses, seed)
        ctrl = make_controller("switching-adaptive", feeder, wl, cfg)
        _, traj = simulate(feeder, ctrl, wl, T, warmup=warmup)
        E.append(traj["V"][warmup:] - 1.0)
        Q.append(traj["Q"][warmup:])
    return np.concatenate(E), np.concatenate(Q)


def train_policy(feeder, E, Q, hidden=32, epochs=300, lr=1e-3, seed=0):
    """Behavior-clone the switching controller's local feedback. Returns (policy, mse)."""
    torch.manual_seed(seed)
    n = len(feeder.buses)
    policy = Policy(n, hidden=hidden)
    e = torch.tensor(E, dtype=torch.float32)
    q = torch.tensor(Q, dtype=torch.float32)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(policy(e), q)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse = float(lossf(policy(e), q))
    return policy, mse


def train_policy_safe(feeder, E, Q, dc_buses, p_center, p_radius, e_radius,
                      hidden=32, epochs=600, lr=1e-3, lam=50.0, margin=0.045,
                      batch_box=2048, seed=0):
    """Clone the switching controller AND shrink the worst-case one-step band excursion.

    Each epoch samples (e, p) uniformly from the operating box and adds a hinge penalty on
    |e^+| beyond a margin inside the +/-5% band, backpropagating through the one-step physics.
    This pushes the network's worst-case output away from the band edge so that CROWN, which
    adds a relaxation gap on top of the true worst case, can still certify the band. It keeps the
    behavior-cloning term so the policy stays close to the paper's controller. Returns
    (policy, onestep, clone_mse).
    """
    torch.manual_seed(seed)
    n = len(feeder.buses)
    from src.load import nominal_injections
    _, q_base = nominal_injections(feeder)
    policy = Policy(n, hidden=hidden)
    onestep = OneStep(policy, feeder.R, feeder.X, q_base)
    e = torch.tensor(E, dtype=torch.float32)
    q = torch.tensor(Q, dtype=torch.float32)
    pc = torch.tensor(p_center, dtype=torch.float32)
    pr = torch.tensor(p_radius, dtype=torch.float32)
    er = torch.tensor(e_radius, dtype=torch.float32)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    mse = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        clone = mse(policy(e), q)
        eb = (torch.rand(batch_box, n) * 2 - 1) * er
        pb = pc + (torch.rand(batch_box, n) * 2 - 1) * pr
        eplus = onestep(eb, pb)
        safety = torch.relu(eplus.abs() - margin).mean()
        (clone + lam * safety).backward()
        opt.step()
    with torch.no_grad():
        clone_mse = float(mse(policy(e), q))
    return policy, onestep, clone_mse


def operating_box(cfg, feeder, dc_buses, e_radius, mode=None):
    """Operating box in (e, p).

    e is centered at 0 with per-bus radius e_radius. p is centered at the nominal injection with
    the data-center buses ranging over their load: the full compute-to-comm swing when mode is
    None, or a single mode +/- the intra-mode fluctuation wbar when mode is given.
    """
    p_base, _ = nominal_injections(feeder)
    n = len(feeder.buses)
    w = cfg["workload"]
    p_center = p_base.copy()
    p_radius = np.zeros(n)
    for bus in dc_buses:
        i = feeder.idx[bus]
        if mode is None:
            lo = -(w["compute_pu"] + w["wbar"])
            hi = -(w["comm_pu"] - w["wbar"])
            p_center[i] = p_base[i] - 0.5 * (w["compute_pu"] + w["comm_pu"])
            p_radius[i] = 0.5 * (hi - lo)
        else:
            pbar = w["compute_pu"] if mode == 1 else w["comm_pu"]
            p_center[i] = p_base[i] - pbar
            p_radius[i] = w["wbar"]
    e_center = np.zeros(n)
    e_rad = np.full(n, e_radius)
    return e_center, e_rad, p_center, p_radius


def mode_equilibrium(onestep, p_center, n, iters=300):
    """Fixed point e* of e -> OneStep(e, p_center) for a fixed load (the mode equilibrium)."""
    e = np.zeros(n)
    p = torch.tensor(p_center, dtype=torch.float32).unsqueeze(0)
    for _ in range(iters):
        with torch.no_grad():
            e = onestep(torch.tensor(e, dtype=torch.float32).unsqueeze(0), p).numpy()[0]
    return e


def save_policy(path, policy):
    torch.save(policy.state_dict(), path)


def load_policy(path, n, hidden=32):
    policy = Policy(n, hidden=hidden)
    policy.load_state_dict(torch.load(path, weights_only=True))
    policy.eval()
    return policy


def load_center(cfg, feeder, dc_buses, mode):
    """Center of the load box: nominal injections plus the data-center load in the given mode."""
    p_base, q_base = nominal_injections(feeder)
    w = cfg["workload"]
    pbar = w["compute_pu"] if mode == 1 else w["comm_pu"]
    p = p_base.copy()
    for bus in dc_buses:
        p[feeder.idx[bus]] = p_base[feeder.idx[bus]] - pbar
    return p, q_base
