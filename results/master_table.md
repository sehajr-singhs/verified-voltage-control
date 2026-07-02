# Master results

All numbers trace to a per-seed JSON under `results/` written by an experiment script.


## E1/E2 control (mean over 5 seeds)

| Exp | Scenario | Controller | max \|v-1\| | violation % | reactive effort |
|---|---|---|---|---|---|
| E2 | single DC (bus 22) | fixed droop | 0.0587 | 0.168 | 13.16 |
| E2 | single DC (bus 22) | switching (oracle) | 0.0429 | 0.000 | 0.00 |
| E2 | single DC (bus 22) | switching (adaptive, ours) | 0.0439 | 0.000 | 1.53 |
| E2 | two DC (buses 22, 25) | fixed droop | 0.0611 | 0.272 | 34.64 |
| E2 | two DC (buses 22, 25) | switching (oracle) | 0.0441 | 0.000 | 0.00 |
| E2 | two DC (buses 22, 25) | switching (adaptive, ours) | 0.0538 | 0.006 | 8.21 |

## E3 exact certificate (published controller)

- gain condition 0 < K < 2X^-1 satisfied: **True**
- rho(I-XK) = **0.99941**, C = 48.82
- every trajectory inside the certified envelope: **True** (uses 1.26% of it)

## E4 neural controller + CROWN

- behavior-cloning MSE = 1.02e-02
- max certified operating radius: IBP = None, CROWN = 0.04, alpha-CROWN = 0.04
- at r = 0.04: band certified = **True**, certified bound is 3.05x the empirical worst case
- certified one-step contraction gamma: compute = 0.614, comm = 0.592

## E5 verification boundary

- certified boundary: IBP = None, CROWN = 0.041, alpha-CROWN = 0.042
- empirical boundary (true worst case leaves band): 0.149
