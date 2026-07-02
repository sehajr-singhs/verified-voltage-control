"""Assemble the master results table from every experiment's saved JSON."""

from __future__ import annotations

import json
import os

from src.io_utils import RESULTS, save_json


def _load(*p):
    path = os.path.join(RESULTS, *p)
    return json.load(open(path)) if os.path.exists(path) else None


def build():
    rows = []

    for tag, scen in [("dc22", "single DC (bus 22)"), ("dc22-25", "two DC (buses 22, 25)")]:
        s = _load("e2", f"summary_{tag}.json")
        if not s:
            continue
        for c, disp in [("fixed", "fixed droop"),
                        ("switching-oracle", "switching (oracle)"),
                        ("switching-adaptive", "switching (adaptive, ours)")]:
            m = s[c]
            rows.append({
                "experiment": "E2", "scenario": scen, "controller": disp,
                "max_dev": round(m["max_dev"]["mean"], 4),
                "violation_pct": round(100 * m["violation_frac"]["mean"], 3),
                "reactive_effort": round(m["dq_total"]["mean"], 2),
            })

    e3 = _load("e3", "certificate_dc22.json")
    e4 = _load("e4", "crown_certificate.json")
    e5 = _load("e5", "boundary.json")

    summary = {
        "E2_table": rows,
        "E3_exact_certificate": None,
        "E4_neural_crown": None,
        "E5_boundary": None,
    }
    if e3:
        summary["E3_exact_certificate"] = {
            "gain_condition_satisfied": e3["gain_condition"]["satisfied"],
            "rho_I_minus_XK": round(e3["contraction"]["rho"], 5),
            "C": round(e3["contraction"]["C"], 2),
            "all_trajectories_inside_envelope": e3["all_buses_inside_envelope"],
            "envelope_utilization_pct": round(100 * e3["envelope_utilization_max"], 2),
        }
    if e4:
        h = e4["headline_alpha_crown"]
        summary["E4_neural_crown"] = {
            "clone_mse": e4["clone_mse"],
            "max_certified_radius": e4["max_certified_radius"],
            "headline_radius": e4["headline_radius"],
            "headline_band_certified": h["band_certified"],
            "headline_tightness_vs_empirical": round(h["tightness_vs_empirical"], 2),
            "contraction_gamma_compute": round(e4["contraction"]["compute"]["sweep"][0]["gamma"], 3),
            "contraction_gamma_comm": round(e4["contraction"]["comm"]["sweep"][0]["gamma"], 3),
        }
    if e5:
        summary["E5_boundary"] = {
            "certified_boundary": e5["certified_boundary"],
            "empirical_boundary": e5["empirical_boundary"],
        }

    save_json(os.path.join(RESULTS, "master_table.json"), summary)
    write_markdown(summary)
    return summary


def write_markdown(summary):
    lines = ["# Master results\n",
             "All numbers trace to a per-seed JSON under `results/` written by an experiment script.\n",
             "\n## E1/E2 control (mean over 5 seeds)\n",
             "| Exp | Scenario | Controller | max \\|v-1\\| | violation % | reactive effort |",
             "|---|---|---|---|---|---|"]
    for r in summary["E2_table"]:
        lines.append(f"| {r['experiment']} | {r['scenario']} | {r['controller']} | "
                     f"{r['max_dev']:.4f} | {r['violation_pct']:.3f} | {r['reactive_effort']:.2f} |")

    e3 = summary["E3_exact_certificate"]
    if e3:
        lines += ["\n## E3 exact certificate (published controller)\n",
                  f"- gain condition 0 < K < 2X^-1 satisfied: **{e3['gain_condition_satisfied']}**",
                  f"- rho(I-XK) = **{e3['rho_I_minus_XK']}**, C = {e3['C']}",
                  f"- every trajectory inside the certified envelope: **{e3['all_trajectories_inside_envelope']}** "
                  f"(uses {e3['envelope_utilization_pct']}% of it)"]
    e4 = summary["E4_neural_crown"]
    if e4:
        lines += ["\n## E4 neural controller + CROWN\n",
                  f"- behavior-cloning MSE = {e4['clone_mse']:.2e}",
                  f"- max certified operating radius: IBP = {e4['max_certified_radius']['IBP']}, "
                  f"CROWN = {e4['max_certified_radius']['CROWN']}, "
                  f"alpha-CROWN = {e4['max_certified_radius']['alpha-CROWN']}",
                  f"- at r = {e4['headline_radius']}: band certified = **{e4['headline_band_certified']}**, "
                  f"certified bound is {e4['headline_tightness_vs_empirical']}x the empirical worst case",
                  f"- certified one-step contraction gamma: compute = {e4['contraction_gamma_compute']}, "
                  f"comm = {e4['contraction_gamma_comm']}"]
    e5 = summary["E5_boundary"]
    if e5:
        cb = e5["certified_boundary"]
        lines += ["\n## E5 verification boundary\n",
                  f"- certified boundary: IBP = {cb['IBP']}, CROWN = {cb['CROWN']:.3f}, "
                  f"alpha-CROWN = {cb['alpha-CROWN']:.3f}",
                  f"- empirical boundary (true worst case leaves band): {e5['empirical_boundary']:.3f}"]

    with open(os.path.join(RESULTS, "master_table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    build()
    print("wrote results/master_table.json and results/master_table.md")
