"""CROWN bound propagation for the neural controller (auto_LiRPA).

Certifies the one-step closed-loop deviation map e^+ = OneStep(e, p) over an operating box in
(e, p): every method returns sound per-bus lower/upper bounds on e^+ that hold for ALL inputs in
the box. IBP is the cheap interval pass, CROWN the backward linear relaxation, alpha-CROWN the
optimized version; tightness improves in that order and we report the gap to a large empirical
sample, which is the certified-versus-empirical story.
"""

from __future__ import annotations

import numpy as np
import torch

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

METHODS = ["IBP", "CROWN", "alpha-CROWN"]


def _box_tensor(center, radius):
    c = torch.tensor(center, dtype=torch.float32).unsqueeze(0)
    r = torch.tensor(radius, dtype=torch.float32).unsqueeze(0)
    return c, c - r, c + r


def make_bounded(onestep, e_center, p_center):
    e0 = torch.tensor(e_center, dtype=torch.float32).unsqueeze(0)
    p0 = torch.tensor(p_center, dtype=torch.float32).unsqueeze(0)
    return BoundedModule(onestep, (e0, p0), verbose=False)


def certify(bm, e_center, e_radius, p_center, p_radius, methods=METHODS):
    """Return {method: (lb[n], ub[n])} certified per-bus bounds on e^+ over the box."""
    e0, eL, eU = _box_tensor(e_center, e_radius)
    p0, pL, pU = _box_tensor(p_center, p_radius)
    be = BoundedTensor(e0, PerturbationLpNorm(norm=float("inf"), x_L=eL, x_U=eU))
    bp = BoundedTensor(p0, PerturbationLpNorm(norm=float("inf"), x_L=pL, x_U=pU))
    out = {}
    for m in methods:
        lb, ub = bm.compute_bounds(x=(be, bp), method=m)
        out[m] = (lb.detach().numpy()[0], ub.detach().numpy()[0])
    return out


def empirical_sweep(onestep, e_center, e_radius, p_center, p_radius, n_samples=100000, seed=0):
    """Sample the box and return the empirical per-bus min/max of e^+ (a lower bound on the true range)."""
    rng = np.random.default_rng(seed)
    n = len(e_center)
    e = e_center + rng.uniform(-1, 1, (n_samples, n)) * e_radius
    p = p_center + rng.uniform(-1, 1, (n_samples, n)) * p_radius
    with torch.no_grad():
        out = onestep(torch.tensor(e, dtype=torch.float32),
                      torch.tensor(p, dtype=torch.float32)).numpy()
    return out.min(axis=0), out.max(axis=0)


def band_certified(lb, ub, band=0.05):
    """True iff the certified box [lb, ub] is inside the +/-band for every bus."""
    return bool((lb >= -band).all() and (ub <= band).all())


def tightness(cert_lb, cert_ub, emp_lb, emp_ub):
    """Relative looseness of a certified box vs the empirical range, averaged over buses."""
    cert_w = cert_ub - cert_lb
    emp_w = np.maximum(emp_ub - emp_lb, 1e-9)
    return float(np.mean(cert_w / emp_w))
