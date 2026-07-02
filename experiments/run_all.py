"""Driver: run every experiment end to end, then figures and the master table.

--quick runs a fast smoke test (1-2 seeds, short horizons, fewer training epochs) that touches
every stage in minutes on CPU. --full reproduces the reported numbers (5 seeds, full horizons).
"""

from __future__ import annotations

import argparse

from experiments import (e1_fixed_failure, e2_switching_fix, e3_exact_cert,
                         e4_neural_crown, e5_boundary, make_figures, make_summary)


def main(quick=False):
    e1_fixed_failure.main(quick=quick)
    e2_switching_fix.main(quick=quick)
    e3_exact_cert.main(quick=quick)
    e4_neural_crown.main(quick=quick)
    e5_boundary.main(quick=quick)
    make_figures.main()
    make_summary.build()
    print("\nAll experiments complete. Figures in static/figures/, table in results/master_table.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick and not args.full)
