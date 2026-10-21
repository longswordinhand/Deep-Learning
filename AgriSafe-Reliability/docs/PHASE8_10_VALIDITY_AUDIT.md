# Phase 8-10: Post-hoc heuristic audit and validity-first repair

## Question
Can the calibration reserve be justified without tuning its coefficient on all three field tasks?

## Phase 8: leave-one-task-out coefficient sensitivity
Candidate rule: n_cal = max(ceil(1/alpha), ceil(rho B)), alpha=0.1, rho in {0.05,0.075,0.10,0.125,0.15,0.20,0.25}.
For each held-out task, rho was selected using only the other two tasks subject to mean coverage >= 0.90 in every training task x budget cell.

Result: the heuristic does not fully generalize. Holding out Apple, the two corn tasks selected rho=0.05; Apple B=160 then achieved only 0.8648 mean coverage. Therefore a tuned proportional reserve is not a defensible cross-task method claim.

## Mechanism diagnosis
The earlier joint-budget protocol selected entropy-based adaptation samples first and drew calibration labels from the remaining pool. This depletes high-entropy examples from calibration and can destroy exchangeability between the calibration sample and future field test examples.

## Phase 9: calibration-first partition
Protocol repair: randomly reserve target calibration examples before any active acquisition. Active adaptation selection is then restricted to the remaining pool. With n_cal=10, this materially improved the problematic Apple B=160 cell from 0.8648 to 0.8931 mean coverage, supporting the selection-bias diagnosis, but two cells remained slightly below 0.90 (PlantWild B=40: 0.8943; Apple B=160: 0.8931).

## Phase 10: alpha-derived validity floor
For standard split conformal at alpha=0.1, n_cal=19 is the smallest calibration size for which the nominal conformal order statistic is not forced to be the maximum nonconformity score. We therefore tested the dataset-independent floor

n_cal* = ceil((2-alpha)/alpha) = 19,

with calibration-first partition and all remaining labels used for entropy-selected adaptation.

Across 3 field targets x 4 total budgets = 12 cells:
- 12/12 achieved mean coverage >= 0.90.
- Minimum mean coverage: 0.9404.
- Mean coverage across cells: 0.9512.

The cost is explicit. At B=20 only one label remains for adaptation, so predictive accuracy is much lower than with n_cal=10. The validity floor should therefore be interpreted as a conservative operating point, not a universally optimal allocation.

## Current claim boundary
Supported:
1. Active acquisition before calibration can bias the calibration pool and degrade field coverage.
2. Calibration-first partition is the correct protocol for joint active-adaptation + conformal-calibration budgeting.
3. A dataset-independent alpha-derived calibration floor of 19 produced >=90% mean coverage in all 12 tested task-budget cells.
4. Hard validity consumes substantial annotation budget at low total budgets, revealing a real feasibility tradeoff.

Not supported:
- rho=0.1 is universally optimal.
- n_cal=19 is a universal minimum for all datasets or all notions of conditional validity.
- CAR-P has proven cross-task optimality.

## Decision
REVISE the method identity away from post-hoc proportional CAR-P toward a validity-first calibration-first budgeting framework with two operating regimes:
- lean regime: small calibration reserve for higher accuracy but weaker empirical coverage stability;
- conservative regime: alpha-derived validity floor before adaptation.
