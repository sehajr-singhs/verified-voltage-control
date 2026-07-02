"""Shared runner: sweep controllers over seeds on a data-center scenario, save results."""

from __future__ import annotations

import os

import numpy as np

from src.config import load_config, make_controller, make_workload
from src.grid import Feeder
from src.io_utils import mean_std, results_dir, save_json, save_npz
from src.sim import simulate

METRIC_KEYS = ["max_dev", "violation_frac", "n_violations", "dq_total", "mean_abs_q",
               "v_min", "v_max", "startup_max_dev"]


def run_scenario(exp_name, dc_buses, controllers, cfg=None, seeds=None, T=None,
                 save_traj_seed=0):
    """Run each controller over all seeds; write per-seed JSON and one trajectory npz each.

    Returns a dict {controller_name: {metric: {mean,std,values}}} aggregated over seeds.
    """
    cfg = cfg or load_config()
    seeds = seeds if seeds is not None else cfg["sim"]["seeds"]
    T = T or cfg["sim"]["T"]
    warmup = cfg["sim"]["warmup"]
    feeder = Feeder()
    outdir = results_dir(exp_name)

    summary = {}
    for cname in controllers:
        per_seed = {k: [] for k in METRIC_KEYS}
        for seed in seeds:
            wl = make_workload(cfg, dc_buses, seed)
            ctrl = make_controller(cname, feeder, wl, cfg)
            metrics, traj = simulate(feeder, ctrl, wl, T, warmup=warmup)
            metrics["seed"] = seed
            metrics["controller"] = cname
            metrics["dc_buses"] = dc_buses
            save_json(os.path.join(outdir, f"{cname}_dc{'-'.join(map(str, dc_buses))}_seed{seed}.json"),
                      metrics)
            for k in METRIC_KEYS:
                per_seed[k].append(metrics[k])
            if seed == save_traj_seed:
                save_npz(os.path.join(outdir, f"traj_{cname}_dc{'-'.join(map(str, dc_buses))}.npz"),
                         V=traj["V"], Q=traj["Q"], REF=traj["REF"], MODE=traj["MODE"])
        summary[cname] = {k: mean_std(v) for k, v in per_seed.items()}
    save_json(os.path.join(outdir, f"summary_dc{'-'.join(map(str, dc_buses))}.json"), summary)
    return summary


def print_summary(title, summary):
    print(f"\n=== {title} ===")
    print(f"{'controller':20s} {'viol%':>7s} {'max_dev':>8s} {'dq_total':>9s} {'v_min':>7s} {'v_max':>7s}")
    for cname, s in summary.items():
        print(f"{cname:20s} {100*s['violation_frac']['mean']:7.2f} {s['max_dev']['mean']:8.4f} "
              f"{s['dq_total']['mean']:9.2f} {s['v_min']['mean']:7.4f} {s['v_max']['mean']:7.4f}")
