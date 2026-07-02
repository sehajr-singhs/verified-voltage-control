"""Regenerate every figure from saved results JSON/npz (no re-simulation, no retraining)."""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.io_utils import FIGURES, RESULTS

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.bbox": "tight",
})
BAND = 0.05
C_FIX, C_ORC, C_ADP = "#c1352e", "#2a8a3e", "#1f6fb2"
C_IBP, C_CROWN, C_ACROWN, C_EMP = "#9a9a9a", "#e08a1e", "#1f6fb2", "#1a1a1a"


def _load_json(*parts):
    p = os.path.join(RESULTS, *parts)
    return json.load(open(p)) if os.path.exists(p) else None


def _load_npz(*parts):
    p = os.path.join(RESULTS, *parts)
    return np.load(p) if os.path.exists(p) else None


def _band(ax, span):
    ax.axhline(1 + BAND, ls="--", lw=1, color=C_FIX, alpha=0.7)
    ax.axhline(1 - BAND, ls="--", lw=1, color=C_FIX, alpha=0.7)
    ax.axhspan(1 - BAND, 1 + BAND, color=C_FIX, alpha=0.04)


def _mode_shading(ax, mode, span):
    m = mode[span]
    t = np.arange(len(m))
    inc = np.where(np.diff(np.concatenate([[0], m])) != 0)[0]
    edges = list(inc) + [len(m)]
    prev = 0
    for e in edges:
        if m[prev] == 1:
            ax.axvspan(prev, e, color="#888888", alpha=0.08)
        prev = e


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name))
    plt.close(fig)
    print("wrote", name)


def fig_e1():
    tr = _load_npz("e1", "traj_fixed_dc22.npz")
    if tr is None:
        return
    V, MODE = tr["V"], tr["MODE"]
    j = 20  # matrix index of bus 22 (buses 2..33 -> index 20)
    span = slice(300, 620)
    fig, ax = plt.subplots(figsize=(7, 3.1))
    _band(ax, span)
    _mode_shading(ax, MODE, span)
    ax.plot(V[span, j], color=C_FIX, lw=1.4)
    ax.set_ylabel("bus-22 voltage (p.u.)")
    ax.set_xlabel("control step")
    ax.set_title("E1  fixed-reference droop overshoots the +5% band at every transition")
    save(fig, "e1_fixed_trace.png")


def fig_e2_traces():
    trs = {c: _load_npz("e2", f"traj_{c}_dc22.npz")
           for c in ["fixed", "switching-oracle", "switching-adaptive"]}
    if any(v is None for v in trs.values()):
        return
    j = 20
    span = slice(300, 620)
    fig, ax = plt.subplots(figsize=(7, 3.3))
    _band(ax, span)
    _mode_shading(ax, trs["fixed"]["MODE"], span)
    ax.plot(trs["fixed"]["V"][span, j], color=C_FIX, lw=1.3, label="fixed droop")
    ax.plot(trs["switching-adaptive"]["V"][span, j], color=C_ADP, lw=1.5,
            label="switching (adaptive, ours)")
    ax.plot(trs["switching-oracle"]["V"][span, j], color=C_ORC, lw=1.1, ls=":",
            label="switching (oracle)")
    ax.set_ylabel("bus-22 voltage (p.u.)")
    ax.set_xlabel("control step")
    ax.set_title("E2  switching reference holds the band")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "e2_traces.png")


def fig_e2_effort():
    labels = ["fixed", "switching-oracle", "switching-adaptive"]
    disp = ["fixed", "oracle", "adaptive"]
    colors = [C_FIX, C_ORC, C_ADP]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, tag, title in [(axes[0], "dc22", "single DC (bus 22)"),
                           (axes[1], "dc22-25", "two DC (buses 22, 25)")]:
        s = _load_json("e2", f"summary_{tag}.json")
        if s is None:
            continue
        vals = [s[c]["dq_total"]["mean"] for c in labels]
        ax.bar(disp, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("cumulative reactive action  $\\sum|\\Delta q|$")
    fig.suptitle("E2  switching reference cuts reactive effort by roughly an order of magnitude",
                 y=1.04)
    fig.tight_layout()
    save(fig, "e2_effort.png")


def fig_e3_envelope():
    tr = _load_npz("e3", "traj_linear_dc22.npz")
    cert = _load_json("e3", "certificate_dc22.json")
    if tr is None or cert is None:
        return
    buses, env, emax = tr["buses"], tr["env"], tr["emax"]
    order = np.argsort(buses)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.semilogy(buses[order], env[order], "-o", ms=3, color=C_CROWN,
                label="certified worst-case envelope (Theorem 1)")
    ax.semilogy(buses[order], np.maximum(emax[order], 1e-6), "-o", ms=3, color=C_EMP,
                label="empirical max $|v-v^{ref}|$")
    ax.set_xlabel("bus number")
    ax.set_ylabel("steady-state $|v-v^{ref}|$ (p.u.)")
    ax.set_title("E3  the exact certificate contains every trajectory (and is loose on the slow mode)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "e3_envelope.png")


def fig_e4_tightness():
    d = _load_json("e4", "crown_certificate.json")
    if d is None:
        return
    r = [row["radius"] for row in d["sweep"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for meth, col in [("IBP", C_IBP), ("CROWN", C_CROWN), ("alpha-CROWN", C_ACROWN)]:
        ub = [row[meth]["ub_max"] for row in d["sweep"]]
        lb = [row[meth]["lb_min"] for row in d["sweep"]]
        ax.plot(r, ub, "-o", ms=3, color=col, label=f"{meth} certified bound")
        ax.plot(r, lb, "-o", ms=3, color=col)
    emp = [row["empirical_e_plus_max"] for row in d["sweep"]]
    ax.plot(r, emp, "--s", ms=3, color=C_EMP, label="empirical worst case")
    ax.plot(r, [-e for e in emp], "--s", ms=3, color=C_EMP)
    ax.axhline(BAND, ls="--", lw=1, color=C_FIX)
    ax.axhline(-BAND, ls="--", lw=1, color=C_FIX)
    ax.axhspan(-BAND, BAND, color=C_FIX, alpha=0.04)
    bnd = _load_json("e5", "boundary.json")
    if bnd and bnd["certified_boundary"].get("alpha-CROWN"):
        rc = bnd["certified_boundary"]["alpha-CROWN"]
        ax.axvline(rc, color=C_ACROWN, lw=1, ls=":", alpha=0.7)
        ax.text(rc, 0.104, f" certified\n boundary\n {rc:.3f}", fontsize=8, color=C_ACROWN, va="top")
    ibp_hi = max(row["IBP"]["ub_max"] for row in d["sweep"])
    ax.text(r[0], 0.118, f"IBP bound reaches $\\pm${ibp_hi:.1f} (off-scale, useless)",
            fontsize=8.5, color=C_IBP)
    ax.set_ylim(-0.13, 0.13)
    ax.set_xlabel("operating-box radius (p.u. voltage deviation)")
    ax.set_ylabel("certified next-step deviation $e^+$")
    ax.set_title("E4  CROWN certifies the band where IBP cannot; bound tracks the empirical worst case")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower left")
    save(fig, "e4_tightness.png")


def fig_e5_boundary():
    d = _load_json("e5", "boundary.json")
    if d is None:
        return
    b = d["certified_boundary"]
    methods = ["IBP", "CROWN", "alpha-CROWN"]
    vals = [b[m] if b[m] else 0.0 for m in methods]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.barh(methods, vals, color=[C_IBP, C_CROWN, C_ACROWN])
    ax.axvline(d["empirical_boundary"], color=C_EMP, ls="--",
               label=f"empirical boundary = {d['empirical_boundary']:.3f}")
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.3f}" if v else "  none", va="center", fontsize=9)
    ax.set_xlabel("largest certified operating radius with +/-5% proven (p.u.)")
    ax.set_title("E5  certified boundary per method vs the true (empirical) boundary")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "e5_boundary.png")


def main():
    fig_e1()
    fig_e2_traces()
    fig_e2_effort()
    fig_e3_envelope()
    fig_e4_tightness()
    fig_e5_boundary()


if __name__ == "__main__":
    main()
