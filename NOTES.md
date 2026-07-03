# Notes: modeling choices and where this artifact departs from the paper

The paper is ground truth. This file records every place where the implementation makes a choice
the paper does not pin down, or where the reproduction differs from the paper, so nothing is
hidden. Nothing here contradicts the paper's equations; these are the free parameters and the
honest simplifications.

## Verification tooling

Prof. Cui pointed to the IBM `CROWN-Robustness-Certification` repo. That repo is the original 2019
CROWN implementation, built for TensorFlow 1 and image classifiers. Its README states it is tested
with "python3 and TensorFlow v1.8 and v1.10". We attempted the install on the Python 3.12 build
machine and it fails at the dependency, with the concrete error

    ERROR: Could not find a version that satisfies the requirement tensorflow==1.15.0
           (from versions: 2.16.0rc0, ..., 2.21.0)
    ERROR: No matching distribution found for tensorflow==1.15.0

because no TensorFlow 1.x wheel exists for Python 3.12 (the oldest published wheel is TF 2.16). The
repo therefore cannot be run as-is on a current Colab. We use **auto_LiRPA**
(github.com/Verified-Intelligence/auto_LiRPA) instead: it is the same CROWN algorithm family
(Zhang et al.), the maintained PyTorch successor, pip-installable, and CPU-fine, and it is what
the neural-network-verification.com tutorial teaches. The IBM repo is cited as the original.
auto_LiRPA's package metadata pins Python `~=3.11`, but the code runs on 3.12; we install it with
`--ignore-requires-python` (documented in `requirements.txt`).

## LinDistFlow scaling

The paper writes v = R p + X q + 1 with R, X positive definite. We build R, X as the path-overlap
(common-ancestor) sensitivity matrices from the Baran-Wu 33-bus branch impedances, with a factor 1
so that v is the voltage **magnitude** in per unit (the +/-5% band is stated on the magnitude).
This reproduces the textbook 33-bus profile (min ~0.913 p.u. at bus 18 before compensation). The
squared-voltage form would carry a factor 2; near 1 p.u. the two differ only by that factor, and
it is absorbed into the gain we choose, so the Theorem 1 conditions are unaffected.

## Load model

- Baran-Wu nominal bus loads are scaled by `LOAD_SCALE = 0.5` (in `src/load.py`) so the base
  feeder sits inside +/-5% before the data center switches. The unscaled 33-bus is stressed enough
  that a few far laterals (notably the bus-30 lateral, 200 kW / 600 kVar) violate at base load,
  which would mask the data-center effect we study. Topology, R/X, and the control math are
  unchanged; only the load magnitudes are scaled.
- The paper drives the data center with a measured ~45-minute DGX/H200 QLoRA power trace. We do not
  have that trace, so the workload is a **modeled** periodic square wave between the two mode levels
  with uniform intra-mode fluctuation. The switching structure, not the exact waveform, is what the
  controller and certificate depend on.
- Data-center magnitudes (compute 0.40 p.u., comm 0.03 p.u. on the 10 MVA base), phase durations,
  and the intra-mode bound wbar are chosen so the fixed-reference baseline breaches +/-5% while the
  switching reference holds it, which is the regime the paper studies. They are in
  `configs/scenario.yaml`.

## Controller

- The droop uses a deadband, a per-step rate limit (the paper's reactive-power saturation), and an
  absolute reactive cap. Values are in the config.
- Section III-C sign selection (Eq. 17) is implemented with **hysteresis** (a margin band around
  the bias), and the amplitude/bias estimates are clipped to physical ranges. Without hysteresis
  the sign chatters when the voltage sits near the bias, which flips the reference by 2*amplitude
  every step and destabilizes the local estimator. This is a guard on the bare equation, noted here
  and in `src/control.py`.
- Proposition 3's bias update has fixed point b* = 1 because the integral droop drives each mode's
  steady-state voltage to its reference; we use that fixed point for the model-based oracle and let
  the adaptive controller estimate it from measurements.

## Exact certificate (E3)

Theorem 1's steady-state bound is realized two ways: the scalar (C, eps) norm bound (reported, and
loose), and the tight per-bus box bound sum_n |A^n| dbar. The latter contains every trajectory but
is conservative on the weakly-controllable slow mode where rho(I-XK) ~ 0.9994, because a diagonal
decentralized gain leaves that mode near the unit circle. This looseness is real and is the stated
motivation for bound propagation on the neural controller.

## Neural certificate (E4/E5)

- The policy is a full-vector MLP e -> q, not a per-bus shared map, which keeps the composite a
  clean Linear-ReLU network for CROWN. It is trained by behavior cloning the switching controller
  plus a safety penalty on the sampled worst-case one-step band excursion. The safety penalty is
  the optional "safe training" stretch the assignment allows; pure cloning certifies only a tiny
  radius because the cloned policy's own worst-case corner sits at the band edge.
- The certified property is one-step band invariance and one-step contraction over an operating
  box, sound for ALL inputs in the box. It is not a multi-step or global guarantee.

## Upstream link

No public code release exists for arXiv:2603.15588 as of July 2026 (checked github.com/Wenqi-Cui).
We do not fabricate a fork. The assignment suggested forking `Wenqi-Cui/Voltage-Control`, but that
repository does not exist (GitHub returns 404 as of July 2026), so there is nothing to fork. Her
actual related public repos are `Wenqi-Cui/Lyapunov-Regularized-RL` and `Wenqi-Cui/MIMO-Neural-PI`
(the Neural-PI and Lyapunov-regularized-RL control line this paper descends from). We fork the
first into `sehajr-singhs/Lyapunov-Regularized-RL` as the honest upstream lineage link, because it
is her closest certified-stability learned-control code and this project adds verification to exactly
that kind of controller. The main repo here stands alone on the paper and does not claim to fork this
paper's (nonexistent) code. Everything is released under the MIT license.
