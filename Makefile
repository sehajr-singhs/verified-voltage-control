# Verified voltage control for AI-training data centers. CPU-first.
# Every target writes real artifacts to disk; every figure regenerates from saved JSON.

PY ?= python

.PHONY: quick full figures summary test clean

quick:      ## Smoke-test every experiment end to end (1-2 seeds, short horizons), minutes on CPU.
	$(PY) -m experiments.run_all --quick

full:       ## Reproduce the reported numbers: 5 seeds, full horizons, full training.
	$(PY) -m experiments.run_all --full

figures:    ## Regenerate every figure from saved results JSON/npz (no re-simulation, no retraining).
	$(PY) -m experiments.make_figures

summary:    ## Rebuild the master results table from saved JSON.
	$(PY) -m experiments.make_summary

test:       ## Fast self-tests of the grid model, controllers, and Theorem 1 gain condition.
	$(PY) -m pytest -q tests

clean:
	rm -rf results/*/ static/figures/*.png results/master_table.*
