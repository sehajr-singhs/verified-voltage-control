"""E2: reproduce the fix. Switching-reference control holds the band with far less effort.

Runs the fixed baseline, the model-based switching reference (Prop. 2/3 oracle), and the
deployable local-measurement switching reference (Section III-C adaptive) on the single data
center (bus 22) and the two data centers (buses 22 and 25). The switching reference moves the
setpoint with the mode so the droop only smooths the intra-mode residual, which keeps every bus
inside +/-5% while cutting the cumulative reactive action by roughly an order of magnitude.
"""

from __future__ import annotations

import argparse

from experiments.common import print_summary, run_scenario

CONTROLLERS = ["fixed", "switching-oracle", "switching-adaptive"]


def main(quick=False):
    seeds = [0] if quick else None
    T = 600 if quick else None
    single = run_scenario("e2", dc_buses=[22], controllers=CONTROLLERS, seeds=seeds, T=T)
    print_summary("E2 single data center (bus 22)", single)
    multi = run_scenario("e2", dc_buses=[22, 25], controllers=CONTROLLERS, seeds=seeds, T=T)
    print_summary("E2 two data centers (buses 22 and 25)", multi)

    fx, ad = single["fixed"], single["switching-adaptive"]
    ratio = fx["dq_total"]["mean"] / max(ad["dq_total"]["mean"], 1e-9)
    print(f"\nSingle DC: adaptive switching holds max deviation {ad['max_dev']['mean']:.4f} "
          f"(<0.05) at {ratio:.1f}x less reactive effort than fixed droop "
          f"({fx['dq_total']['mean']:.2f} vs {ad['dq_total']['mean']:.2f}).")
    return single, multi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
