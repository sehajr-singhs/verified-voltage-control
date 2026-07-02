"""E3: exact certification of the published switching-reference controller (Theorem 1).

Verifies the gain condition 0 < K < 2 X^-1 on the real feeder reactance matrix, computes the
spectral radius rho(I - XK), the contraction constants (C, eps), and the closed-form per-bus
steady-state error envelope, then runs the ideal linear closed loop (the affine system Theorem 1
describes, with the practical deadband and saturations switched off) and confirms every bus of
every trajectory lives inside the certified envelope. The envelope is sound everywhere and tight
at the well-controlled buses, but conservative on the weakly-controllable slow mode where
rho(A) ~ 0.9994, which is precisely the gap that motivates bound propagation in E4.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from src.certify_exact import certified_envelope
from src.config import gain, load_config, make_workload
from src.control import SwitchingRefOracle
from src.grid import Feeder
from src.io_utils import results_dir, save_json, save_npz
from src.sim import simulate

LINEAR = dict(deadband=0.0, dq_max=1e9, q_cap=1e9)   # the affine system of Theorem 1


def run_case(feeder, cfg, dc_buses, T, warmup):
    wl = make_workload(cfg, dc_buses, seed=0)
    k = gain(feeder, cfg)
    cert = certified_envelope(feeder, wl, k)

    ctrl = SwitchingRefOracle(feeder, wl, k, **LINEAR)
    _, traj = simulate(feeder, ctrl, wl, T, warmup=warmup)
    err = np.abs(traj["V"] - traj["REF"])[warmup:]        # steady-state tracking error
    emax = err.max(axis=0)                                 # empirical worst per bus
    env = np.array(cert["env_per_bus"])
    inside = bool((emax <= env + 1e-9).all())

    j22 = feeder.idx[22]
    cert.update({
        "dc_buses": dc_buses,
        "empirical_max_err_overall": float(emax.max()),
        "all_buses_inside_envelope": inside,
        "envelope_utilization_max": float(np.max(emax / np.maximum(env, 1e-12))),
        "env_at_bus22": float(env[j22]),
        "empirical_err_at_bus22": float(emax[j22]),
        "argmax_env_bus": int(feeder.buses[int(np.argmax(env))]),
    })
    return cert, traj, emax, env


def main(quick=False):
    cfg = load_config()
    feeder = Feeder()
    T = 1500 if quick else 3000
    warmup = cfg["sim"]["warmup"]
    outdir = results_dir("e3")

    for dc_buses in ([22], [22, 25]):
        cert, traj, emax, env = run_case(feeder, cfg, dc_buses, T, warmup)
        tag = "-".join(map(str, dc_buses))
        save_json(os.path.join(outdir, f"certificate_dc{tag}.json"), cert)
        save_npz(os.path.join(outdir, f"traj_linear_dc{tag}.npz"),
                 V=traj["V"], REF=traj["REF"], MODE=traj["MODE"],
                 emax=emax, env=env, buses=np.array(feeder.buses))
        gc = cert["gain_condition"]
        print(f"\n=== E3 exact certificate, DC buses {dc_buses} ===")
        print(f"  gain condition 0<K<2X^-1: {gc['satisfied']} "
              f"(XK eig in [{gc['XK_eig_min']:.4f}, {gc['XK_eig_max']:.4f}])")
        print(f"  rho(I-XK) = {cert['contraction']['rho']:.5f}, C = {cert['contraction']['C']:.2f}")
        print(f"  certified env: bus22={cert['env_at_bus22']:.4f}, "
              f"max={cert['env_max']:.4f} at bus {cert['argmax_env_bus']}")
        print(f"  empirical max|v-ref| = {cert['empirical_max_err_overall']:.5f}; "
              f"all buses inside envelope: {cert['all_buses_inside_envelope']} "
              f"(uses {100*cert['envelope_utilization_max']:.1f}% of it)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
