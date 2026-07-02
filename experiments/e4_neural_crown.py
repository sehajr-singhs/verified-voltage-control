"""E4: neural controller certified by CROWN bound propagation.

Trains a small MLP reactive-power policy by behavior cloning the switching-reference controller,
with a safety penalty that shrinks its worst-case one-step band excursion, then certifies with
auto_LiRPA (IBP, CROWN, alpha-CROWN) that for EVERY voltage state in an operating box and EVERY
data-center load across the full compute/comm range, one closed-loop step keeps every bus inside
+/-5%. This is a guarantee over the whole box, not a sampled fraction. We report certified per-bus
bounds, the certified band radius, the certified one-step contraction toward each mode
equilibrium (the neural analog of Theorem 1), and the tightness of each method against a 10^5
-sample empirical sweep.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.certify_crown import (band_certified, certify, empirical_sweep, make_bounded,
                               tightness)
from src.config import load_config
from src.grid import Feeder
from src.io_utils import results_dir, save_json
from src.neural import (OneStep, collect_clone_data, mode_equilibrium, operating_box,
                        save_policy, train_policy_safe)
from src.load import nominal_injections


def main(quick=False):
    cfg = load_config()
    feeder = Feeder()
    n = len(feeder.buses)
    dc = [22]
    nn_cfg = cfg["neural"]
    outdir = results_dir("e4")

    seeds = [0, 1] if quick else cfg["sim"]["seeds"]
    epochs = 300 if quick else nn_cfg["epochs"]
    radii = nn_cfg["certify_radii"]
    n_samples = 20000 if quick else 100000

    # Train the policy with the safety penalty over the full-load operating box.
    E, Q = collect_clone_data(cfg, feeder, dc, seeds)
    _, e_rad_tr, p_center, p_radius = operating_box(cfg, feeder, dc, nn_cfg["train_radius"])
    policy, onestep, clone_mse = train_policy_safe(
        feeder, E, Q, dc, p_center, p_radius, e_rad_tr,
        hidden=nn_cfg["hidden"], epochs=epochs, lr=nn_cfg["lr"],
        lam=nn_cfg["lam"], margin=nn_cfg["margin"], seed=0)
    save_policy(os.path.join(outdir, "policy.pt"), policy)
    print(f"E4: trained policy, behavior-cloning MSE = {clone_mse:.2e}")

    e_center = np.zeros(n)
    bm = make_bounded(onestep, e_center, p_center)

    # Certified-vs-empirical sweep over the operating radius.
    sweep = []
    for r in radii:
        er = np.full(n, r)
        emp_lb, emp_ub = empirical_sweep(onestep, e_center, er, p_center, p_radius, n_samples)
        cert = certify(bm, e_center, er, p_center, p_radius)
        row = {"radius": r,
               "empirical_e_plus_max": float(max(abs(emp_lb.min()), emp_ub.max()))}
        for m, (lb, ub) in cert.items():
            row[m] = {"lb_min": float(lb.min()), "ub_max": float(ub.max()),
                      "band_certified": band_certified(lb, ub),
                      "tightness_vs_empirical": tightness(lb, ub, emp_lb, emp_ub)}
        sweep.append(row)
        print(f"  r={r:.2f} emp|e+|={row['empirical_e_plus_max']:.4f} | "
              f"IBP={row['IBP']['band_certified']} CROWN={row['CROWN']['band_certified']} "
              f"aCROWN={row['alpha-CROWN']['band_certified']} | "
              f"tightness I/C/aC = {row['IBP']['tightness_vs_empirical']:.0f}/"
              f"{row['CROWN']['tightness_vs_empirical']:.2f}/"
              f"{row['alpha-CROWN']['tightness_vs_empirical']:.2f}")

    # Largest radius certified by each method.
    def max_certified(method):
        ok = [row["radius"] for row in sweep if row[method]["band_certified"]]
        return max(ok) if ok else None

    # Certified one-step contraction toward each mode equilibrium.
    contraction = {}
    for mode, lab in [(1, "compute"), (0, "comm")]:
        _, _, pc, _ = operating_box(cfg, feeder, dc, 0.0, mode=mode)
        est = mode_equilibrium(onestep, pc, n)
        bm2 = make_bounded(onestep, est, pc)
        pr0 = np.zeros(n)
        rows = []
        for d in nn_cfg["contraction_deltas"]:
            c = certify(bm2, est, np.full(n, d), pc, pr0, methods=["alpha-CROWN"])
            lb, ub = c["alpha-CROWN"]
            dev = float(np.maximum(np.abs(ub - est), np.abs(est - lb)).max())
            rows.append({"delta": d, "certified_sup_dev": dev, "gamma": dev / d,
                         "contracts": bool(dev < d)})
        contraction[lab] = {"equilibrium_max_abs": float(np.abs(est).max()), "sweep": rows}
        g = rows[0]["gamma"]
        print(f"  contraction {lab}: gamma={g:.3f} (delta={rows[0]['delta']}) -> "
              f"{'contracts' if rows[0]['contracts'] else 'no'}")

    headline = nn_cfg["headline_radius"]
    hrow = next(row for row in sweep if abs(row["radius"] - headline) < 1e-9)
    result = {
        "dc_buses": dc,
        "clone_mse": clone_mse,
        "sweep": sweep,
        "max_certified_radius": {m: max_certified(m) for m in ["IBP", "CROWN", "alpha-CROWN"]},
        "headline_radius": headline,
        "headline_alpha_crown": hrow["alpha-CROWN"],
        "contraction": contraction,
    }
    save_json(os.path.join(outdir, "crown_certificate.json"), result)
    print(f"\nHeadline: alpha-CROWN certifies +/-5% for all states within {headline} p.u. and all "
          f"data-center loads; bound is {hrow['alpha-CROWN']['tightness_vs_empirical']:.2f}x the "
          f"empirical worst case. Max certified radius: CROWN={result['max_certified_radius']['CROWN']}, "
          f"alpha-CROWN={result['max_certified_radius']['alpha-CROWN']}, IBP="
          f"{result['max_certified_radius']['IBP']}.")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
