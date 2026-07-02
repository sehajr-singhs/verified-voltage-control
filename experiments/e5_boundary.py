"""E5: where verification breaks. Grow the operating box until the certificate fails.

Reuses the E4-trained policy and grows the voltage-state box radius, recording for each method
(IBP, CROWN, alpha-CROWN) the largest radius at which the +/-5% band is still certified, and
comparing to the empirical boundary where the true worst case first leaves the band. A certificate
that holds everywhere it claims and visibly fails past a boundary is more credible than one shown
without edges, and where alpha-CROWN certifies a radius plain CROWN cannot we report that rescue.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from src.certify_crown import band_certified, certify, empirical_sweep, make_bounded
from src.config import load_config
from src.grid import Feeder
from src.io_utils import results_dir, save_json
from src.neural import OneStep, load_policy, operating_box
from src.load import nominal_injections


def band_ok_at(bm, onestep, e_center, r, p_center, p_radius, methods, n_samples):
    n = len(e_center)
    er = np.full(n, r)
    cert = certify(bm, e_center, er, p_center, p_radius, methods=methods)
    emp_lb, emp_ub = empirical_sweep(onestep, e_center, er, p_center, p_radius, n_samples)
    out = {m: band_certified(*cert[m]) for m in methods}
    out["empirical_max"] = float(max(abs(emp_lb.min()), emp_ub.max()))
    out["empirical_safe"] = bool(out["empirical_max"] <= 0.05)
    return out


def bisect_boundary(fn, lo, hi, tol=0.001):
    """Largest radius in [lo, hi] where fn(r) is True (assumes monotone True->False)."""
    if not fn(lo):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if fn(mid):
            lo = mid
        else:
            hi = mid
    return lo


def main(quick=False):
    cfg = load_config()
    feeder = Feeder()
    n = len(feeder.buses)
    dc = [22]
    outdir = results_dir("e5")

    policy_path = os.path.join("results", "e4", "policy.pt")
    if not os.path.exists(policy_path):
        raise SystemExit("Run experiments.e4_neural_crown first (need results/e4/policy.pt).")
    policy = load_policy(policy_path, n, hidden=cfg["neural"]["hidden"])
    _, q_base = nominal_injections(feeder)
    onestep = OneStep(policy, feeder.R, feeder.X, q_base)

    e_center, _, p_center, p_radius = operating_box(cfg, feeder, dc, 0.0)
    bm = make_bounded(onestep, e_center, p_center)
    n_samples = 20000 if quick else 100000
    methods = ["IBP", "CROWN", "alpha-CROWN"]

    grid = np.round(np.arange(0.01, 0.085, 0.005), 4)
    sweep = []
    for r in grid:
        row = {"radius": float(r)}
        row.update(band_ok_at(bm, onestep, e_center, r, p_center, p_radius, methods, n_samples))
        sweep.append(row)
        print(f"  r={r:.3f} IBP={row['IBP']} CROWN={row['CROWN']} aCROWN={row['alpha-CROWN']} "
              f"| emp|e+|={row['empirical_max']:.4f} emp_safe={row['empirical_safe']}")

    # Exact per-method certified boundary and the empirical boundary, by bisection.
    boundaries = {}
    for m in methods:
        boundaries[m] = bisect_boundary(
            lambda r: band_certified(*certify(bm, e_center, np.full(n, r), p_center, p_radius,
                                              methods=[m])[m]), 0.005, 0.09)
    emp_boundary = bisect_boundary(
        lambda r: band_ok_at(bm, onestep, e_center, r, p_center, p_radius,
                             ["IBP"], n_samples)["empirical_safe"], 0.005, 0.15)

    rescue = [row["radius"] for row in sweep if row["alpha-CROWN"] and not row["CROWN"]]
    result = {
        "dc_buses": dc,
        "sweep": sweep,
        "certified_boundary": boundaries,
        "empirical_boundary": emp_boundary,
        "alpha_crown_rescue_radii": rescue,
    }
    save_json(os.path.join(outdir, "boundary.json"), result)

    print(f"\nCertified boundaries (max radius with +/-5% proven): "
          f"IBP={boundaries['IBP']}, CROWN={boundaries['CROWN']:.3f}, "
          f"alpha-CROWN={boundaries['alpha-CROWN']:.3f}.")
    print(f"Empirical boundary (true worst case leaves band): {emp_boundary:.3f}.")
    if rescue:
        print(f"alpha-CROWN certifies radii CROWN cannot: {rescue}.")
    else:
        print("alpha-CROWN and CROWN share the boundary at this grid resolution; "
              f"the alpha-CROWN margin at the boundary is smaller (see JSON).")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
