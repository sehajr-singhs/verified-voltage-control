"""Load the scenario config and build feeders, workloads, and controllers from it."""

from __future__ import annotations

import os
import yaml

from src.grid import Feeder
from src.load import DCWorkload, nominal_injections
from src.control import (FixedDroop, SwitchingRefAdaptive, SwitchingRefOracle,
                         max_stable_gain)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "configs", "scenario.yaml")


def load_config(path=CONFIG_PATH):
    with open(path) as fh:
        return yaml.safe_load(fh)


def gain(feeder, cfg):
    """Scalar droop gain k = gain_frac * (max stable gain), inside 0 < kI < 2 X^-1."""
    return cfg["control"]["gain_frac"] * max_stable_gain(feeder.X)


def make_workload(cfg, dc_buses, seed):
    w = cfg["workload"]
    return DCWorkload(dc_buses, compute_pu=w["compute_pu"], comm_pu=w["comm_pu"],
                      compute_steps=w["compute_steps"], comm_steps=w["comm_steps"],
                      wbar=w["wbar"], seed=seed)


def _droop_kw(cfg):
    c = cfg["control"]
    return dict(deadband=c["deadband"], q_cap=c["q_cap"], dq_max=c["dq_max"])


def make_controller(name, feeder, workload, cfg):
    k = gain(feeder, cfg)
    n = len(feeder.buses)
    if name == "fixed":
        return FixedDroop(n, k, **_droop_kw(cfg))
    if name == "switching-oracle":
        return SwitchingRefOracle(feeder, workload, k, **_droop_kw(cfg))
    if name == "switching-adaptive":
        c = cfg["control"]
        return SwitchingRefAdaptive(n, k, La=c["window"], Lb=c["window"],
                                    eta_b=c["eta_b"], **_droop_kw(cfg))
    raise ValueError(f"unknown controller {name}")
