"""E1: reproduce the failure. Fixed-reference droop on the two-mode data-center load.

The fixed reference at 1 p.u. makes the droop re-fight every mode transition. During the
compute phase it winds up reactive power to hold the sagging voltage near 1, and when the load
drops at the communication transition the accumulated reactive over-corrects and the voltage
overshoots past +5%, then the rate-limited droop slowly unwinds and the cycle repeats. The
result is a persistent transition oscillation that breaches the band and spends large reactive
effort. This mirrors the qualitative shape of the paper's Fig. 3 left column.
"""

from __future__ import annotations

import argparse

from experiments.common import print_summary, run_scenario


def main(quick=False):
    seeds = [0] if quick else None
    T = 600 if quick else None
    summary = run_scenario("e1", dc_buses=[22], controllers=["fixed"], seeds=seeds, T=T)
    print_summary("E1 fixed-reference droop, single data center (bus 22)", summary)
    s = summary["fixed"]
    print(f"\nFixed droop breaches +/-5% on {100*s['violation_frac']['mean']:.2f}% of bus-steps, "
          f"peaks at v_max={s['v_max']['mean']:.4f} p.u., and spends dq_total="
          f"{s['dq_total']['mean']:.2f} p.u. of cumulative reactive action.")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
