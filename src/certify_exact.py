"""Exact certification of the published switching-reference controller (Theorem 1).

The paper's controller is piecewise affine: within a mode the closed loop on the reference
tracking error e_t = v_t - v_t^ref is the linear system

    e_{t+1} = A e_t + d_t,    A = I - X K,    d_t = R dp_t - dv_t^ref,

(derived by substituting the droop update q_{t+1} = q_t - K(v_t - v_t^ref) into the LinDistFlow
map). Theorem 1 asks for 0 < K < 2 X^-1, which is exactly rho(A) < 1, and then bounds the
steady-state tracking error. Because this is a linear system with a box-bounded disturbance, its
safety can be checked exactly, which is the ground truth we later compare the neural certificate
against. Everything here is closed form or a convergent matrix sum, no sampling.

Proposition 2 makes the disturbance the intra-mode fluctuation only: choosing the two mode
references b +/- (1/2) R(pbar(0) - pbar(1)) cancels the mode-transition term R dp_mode, leaving
d_t = R dw_t with |dw_t| <= 2 wbar at the data-center buses.
"""

from __future__ import annotations

import numpy as np


def gain_condition(X, k):
    """Check 0 < k I < 2 X^-1 (Theorem 1). Returns the eigenvalues of X K and the verdict."""
    XK_eig = k * np.linalg.eigvalsh(X)           # eigenvalues of kX (== X K for scalar k)
    return {
        "k": float(k),
        "XK_eig_min": float(XK_eig.min()),
        "XK_eig_max": float(XK_eig.max()),
        "satisfied": bool(XK_eig.min() > 0 and XK_eig.max() < 2.0),
    }


def contraction(X, k):
    """rho(A) and a rigorous (C, eps) with |A^n|_2 <= C eps^n, A = I - kX.

    In the X^-1-weighted inner product A is self-adjoint, so its weighted-norm gain equals its
    spectral radius eps = rho(A) = max_i |1 - k lambda_i(X)|, and converting to the Euclidean
    norm costs C = sqrt(cond(X)).
    """
    lamX = np.linalg.eigvalsh(X)
    eps = float(np.max(np.abs(1.0 - k * lamX)))
    C = float(np.sqrt(lamX.max() / lamX.min()))
    return {"rho": eps, "eps": eps, "C": C}


def abs_neumann_sum(A, tol=1e-10, max_terms=200000):
    """S = sum_{n>=0} |A^n| (elementwise abs), convergent because rho(A) < 1.

    Gives the tight per-bus steady-state bound: |e_inf|_i <= (S dbar)_i for |d_t| <= dbar.
    """
    n = A.shape[0]
    S = np.abs(np.eye(n))
    P = np.eye(n)
    terms = 0
    while terms < max_terms:
        P = P @ A
        Pa = np.abs(P)
        S = S + Pa
        terms += 1
        if Pa.max() < tol:
            break
    return S, terms


def disturbance_bound(feeder, workload):
    """Per-bus bound dbar on |d_t| = |R dw_t| after Prop. 2 cancels the mode transition.

    dw_t is the change in intra-mode fluctuation between steps, |dw| <= 2 wbar at each DC bus.
    """
    dw = np.zeros(len(feeder.buses))
    for bus in workload.dc_buses:
        dw[feeder.idx[bus]] = 2.0 * workload.wbar
    return np.abs(feeder.R) @ dw


def certified_envelope(feeder, workload, k):
    """Full Theorem 1 certificate: gain check, contraction constants, per-bus error envelope."""
    A = np.eye(feeder.X.shape[0]) - k * feeder.X
    cond = gain_condition(feeder.X, k)
    con = contraction(feeder.X, k)
    S, terms = abs_neumann_sum(A)
    dbar = disturbance_bound(feeder, workload)
    env = S @ dbar                                       # per-bus certified |v - v^ref| bound
    # Scalar Theorem 1 bound (loose): limsup ||e||_2 <= (1/(1-eps)) C ||dbar||_2.
    scalar = con["C"] / (1.0 - con["eps"]) * float(np.linalg.norm(dbar))
    return {
        "gain_condition": cond,
        "contraction": con,
        "neumann_terms": int(terms),
        "dbar_max": float(dbar.max()),
        "env_per_bus": env.tolist(),
        "env_max": float(env.max()),
        "scalar_theorem1_bound": scalar,
    }
