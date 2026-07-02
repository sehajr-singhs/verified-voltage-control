# Verified Voltage Control for AI-Training Data Centers

**Project site:** https://sehajr-singhs.github.io/verified-voltage-control · **Paper being verified:** Yan, Joswig-Jones, Zhang, Chen, Cui, "Switching-Reference Voltage Control for Distribution Systems with AI-Training Data Centers," [arXiv:2603.15588](https://arxiv.org/abs/2603.15588), 2026.

AI training workloads switch synchronously between a high-power compute phase and a low-power communication phase, which throws periodic step changes in active power onto the distribution feeder and pushes bus voltages toward and past the plus-or-minus 5 percent band. The paper fixes this with a decentralized reactive-power droop whose reference switches with the workload mode, and it proves the closed loop contracts when the gain satisfies 0 < K < 2X^-1. This repo does two things with that result. It reproduces the controller and certifies the paper's own guarantee exactly on the real IEEE 33-bus feeder, because the published controller is piecewise affine so its safety is a spectral-radius and matrix-sum computation rather than a search. Then it trains a small neural controller in the same setting and certifies it with CROWN-style bound propagation, which proves for every voltage state in an operating box and every data-center load across the full mode range, not a sampled fraction of them, that one closed-loop step keeps every bus inside plus-or-minus 5 percent. The through-line is that the published guarantee is exact because the controller is affine, and bound propagation is what lets that same kind of guarantee survive the jump to a network, for all inputs in the region rather than a sampled subset.

Everything is simulation-only and CPU-scale, runs on a free Colab, and every number below traces to a per-seed JSON a script wrote.

## Headline results

- **The failure is real.** Fixed-reference droop on the two-mode data-center load at bus 22 overshoots to **1.059 p.u.** at every communication transition, breaching plus-5 percent, because it winds up reactive power to hold the sagging compute-phase voltage and then over-corrects when the load drops, and it spends **13.2 p.u.** of cumulative reactive action doing it.
- **The fix is real.** The switching-reference controller holds max deviation to **0.044** (inside the band) with **1.5 p.u.** of reactive action, so it cuts the effort **8.6 times** while staying in band, and the model-based reference cancels the mode disturbance so completely (Prop. 2) that its reactive effort drops to essentially zero.
- **The published guarantee is certified exactly.** On the real feeder the gain condition 0 < K < 2X^-1 holds, rho(I-XK) = **0.99941**, and the closed-form Theorem 1 envelope contains **every** trajectory of the linear closed loop, which uses only **1.3 percent** of it. The envelope is tight at the well-controlled buses and conservative on the slow mode near the unit circle, which is exactly why the neural side needs bound propagation.
- **The neural controller is certified for all inputs in the box.** A small MLP trained in the same setting is certified by CROWN and alpha-CROWN to keep every bus inside plus-or-minus 5 percent for **all** voltage states within **0.04 p.u.** of nominal and **all** data-center loads across the compute-to-comm range, with a certified one-step contraction of **gamma = 0.61**. IBP cannot certify any radius because its bound reaches plus-or-minus 1.3, while the CROWN bound is within **3.05 times** the empirical worst case.
- **The certificate has a visible edge.** Growing the box, CROWN certifies up to radius **0.041** and alpha-CROWN up to **0.042**, so alpha-CROWN rescues a sliver plain CROWN loses, and both stop well before the true empirical boundary at **0.149**, which is the honest conservativeness of sound bound propagation.

## The contrast this project is about

The sibling artifact [adaptive-contact-dynamics](https://github.com/sehajr-singhs/adaptive-contact-dynamics) verifies a neural Lyapunov certificate by **sampling**, and reports that it holds on 90.5 percent of sampled states. That is confidence over a region. The CROWN results here are different in kind, because bound propagation proves the property for **every** input in the defined box at once, with no sampling, which is why the number is a certified radius and not a percentage. The whole point of this repo is that jump from a sampled certificate to a certificate sound for all inputs in the set, and the price of soundness is visible in E5 where the certified radius is a fraction of the true one.

## Results table

| Exp | Scenario | Controller | max \|v-1\| | violation % | reactive effort |
|---|---|---|---|---|---|
| E1/E2 | single DC (bus 22) | fixed droop | 0.0587 | 0.168 | 13.16 |
| E1/E2 | single DC (bus 22) | switching (oracle) | 0.0429 | 0.000 | 0.00 |
| E1/E2 | single DC (bus 22) | **switching (adaptive, ours)** | **0.0439** | **0.000** | **1.53** |
| E1/E2 | two DC (buses 22, 25) | fixed droop | 0.0611 | 0.272 | 34.64 |
| E1/E2 | two DC (buses 22, 25) | switching (oracle) | 0.0441 | 0.000 | 0.00 |
| E1/E2 | two DC (buses 22, 25) | **switching (adaptive, ours)** | **0.0538** | **0.006** | **8.21** |

The two-data-center adaptive case grazes the band at 0.0538 on rare intra-mode peaks (0.006 percent of bus-steps), which we report straight because the harder scenario is where the local estimator is most stretched.

| Certificate | Result |
|---|---|
| E3 gain condition 0 < K < 2X^-1 | satisfied, rho(I-XK) = 0.99941, C = 48.82 |
| E3 all trajectories inside envelope | true, uses 1.3% of it |
| E4 max certified radius (IBP / CROWN / alpha-CROWN) | none / 0.04 / 0.04 |
| E4 certified band at r = 0.04 | true, bound 3.05x the empirical worst case |
| E4 certified one-step contraction gamma (compute / comm) | 0.614 / 0.592 |
| E5 certified boundary (CROWN / alpha-CROWN) vs empirical | 0.041 / 0.042 vs 0.149 |

## Reproduce

```bash
# CPU-only. auto_LiRPA's metadata pins Python 3.11 but the code runs on 3.12.
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.6 scipy==1.18.0 matplotlib==3.11.0 pyyaml==6.0.3 pytest==9.1.1
python -m pip install --ignore-requires-python "git+https://github.com/Verified-Intelligence/auto_LiRPA.git"

make test        # grid model, controllers, Theorem 1 gain condition
make quick       # smoke-test every experiment end to end, minutes on CPU
make full        # the reported numbers: 5 seeds, full horizons, full training
make figures     # regenerate every figure from saved JSON/npz, no re-simulation
make summary     # rebuild the master results table
```

`colab/verify.ipynb` installs the dependencies and runs the core exact and CROWN certification end to end in one notebook, because the assignment should run on a free Colab.

## Repo layout

```
src/grid.py           IEEE 33-bus LinDistFlow feeder (Baran-Wu data), v = R p + X q + 1
src/load.py           nominal loads + two-mode data-center workload
src/control.py        fixed droop, switching-reference oracle (Prop. 2/3), adaptive (Sec. III-C)
src/sim.py            closed-loop simulator and metrics
src/certify_exact.py  Theorem 1: gain condition, (C, eps), per-bus envelope
src/neural.py         MLP policy, one-step physics composite, safety-regularized training
src/certify_crown.py  auto_LiRPA IBP / CROWN / alpha-CROWN + empirical sweep
experiments/          one script per experiment (E1..E5), figures, summary, run_all driver
configs/scenario.yaml every scenario and hyperparameter
results/              per-seed JSON + trajectory npz (every reported number lives here)
static/figures/       matplotlib figures, regenerated from results
tests/                grid and controller self-tests
```

## What this is and is not

This is simulation on a linearized feeder model with a modeled two-mode workload, and it has no hardware. LinDistFlow is a linearization that drops the DistFlow loss terms, the load trace is an idealized square wave rather than the paper's measured DGX trace, and the verified property covers the defined input box and nothing outside it. The exact certificate is sound but conservative on the slow mode, and the neural certificate is one-step band invariance and one-step contraction over an operating box, not a multi-step or global proof. The nominal loads are scaled so the base feeder sits inside the band, which is a modeling choice stated in `NOTES.md`. The full list of departures from the paper is in `NOTES.md`, and the paper is ground truth wherever this repo and the paper disagree.

## References

- Yan, Joswig-Jones, Zhang, Chen, Cui. Switching-Reference Voltage Control for Distribution Systems with AI-Training Data Centers. arXiv:2603.15588, 2026. (the paper being verified)
- Prof. Cui's related public code, cited as the closest ancestors of this control line: [Wenqi-Cui/Lyapunov-Regularized-RL](https://github.com/Wenqi-Cui/Lyapunov-Regularized-RL) and [Wenqi-Cui/MIMO-Neural-PI](https://github.com/Wenqi-Cui/MIMO-Neural-PI). No public code release exists for arXiv:2603.15588, and no `Wenqi-Cui/Voltage-Control` repo exists (checked July 2026), so this repo stands alone and does not fabricate a fork. See `NOTES.md`.
- Zhang et al. Efficient Neural Network Robustness Certification with General Activation Functions (CROWN), NeurIPS 2018. IBM `CROWN-Robustness-Certification` is the original implementation Prof. Cui pointed to.
- Xu, Zhang, et al. auto_LiRPA: automatic linear relaxation based perturbation analysis. github.com/Verified-Intelligence/auto_LiRPA (the maintained CROWN-family engine used here)
- Baran, Wu. Network reconfiguration in distribution systems for loss reduction and load balancing. IEEE Trans. Power Delivery, 1989. (the 33-bus feeder data)

## Citation

```bibtex
@misc{singh2026verifiedvoltage,
  title        = {Verified Voltage Control for AI-Training Data Centers},
  author       = {Sehaj Singh},
  year         = {2026},
  howpublished = {\url{https://github.com/sehajr-singhs/verified-voltage-control}}
}
```

## License

MIT.
