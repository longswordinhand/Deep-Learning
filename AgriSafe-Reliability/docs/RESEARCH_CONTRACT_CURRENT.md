# AgriSafe Journal Extension — Current Research Contract

## Problem anchor
How should a scarce real-field label budget be used to recover both predictive performance and valid uncertainty after lab-to-field distribution shift in plant-disease recognition?

## Paper identity
**Validity-first field-label budgeting under agricultural distribution shift.**
The paper is not a new backbone paper, not a claim that DINOv2 is novel, and not a claim that one active-learning score universally dominates.

## Dominant scientific contribution
A deployment-oriented formulation showing that target labels have two distinct and competing uses under field shift: (i) adapting the predictor and (ii) restoring valid target-domain uncertainty calibration. The useful operating region is therefore an accuracy–coverage–set-efficiency frontier rather than an accuracy-only optimum.

## Supporting contribution C2
Acquisition signals themselves are not distribution-shift invariant. Entropy, margin, least-confidence, diversity, and committee disagreement change rank across domains and label budgets; a rule that helps one field target can be neutral or harmful on another.

## Supporting contribution C3
A calibration-first validity budgeting protocol. Target calibration examples must be randomly reserved before active adaptation selection; otherwise active acquisition can deplete difficult examples from the calibration pool and bias coverage. For 90% split conformal, an alpha-derived conservative calibration floor of n_cal=19 is evaluated as a validity-first operating point, not as a universal optimum.

## Data / task boundary
Three public source-to-field targets are currently in scope:
1. PlantVillage corn -> PlantDoc corn.
2. PlantVillage corn -> primary-clean PlantWild corn.
3. PlantVillage apple -> PlantDoc apple.

PlantWild exact duplicate conflicts are audited and the primary analysis uses the predeclared clean subset. Apple target data contain no exact duplicate conflicts.

## Protocol boundary
- Frozen DINOv2-small representations.
- Source-selected linear-probe regularization only.
- Five-fold path-hash target cross-fitting.
- Candidate target labels are unavailable to acquisition rules until selected.
- Calibration labels are randomly reserved before active acquisition in validity-first experiments.
- Held-out target fold labels are evaluation-only.
- Publication metrics must be computed after pooling held-out predictions across folds, not by averaging fold metrics.

## Claims currently allowed
1. Source-domain predictive and conformal reliability can fail substantially after lab-to-field transfer.
2. The best acquisition signal is domain- and budget-dependent across the tested targets; no tested fixed score universally dominates.
3. Target calibration can restore nominal marginal coverage, but coverage alone may be operationally weak when prediction sets become broad.
4. A fixed target-label budget has competing adaptation and calibration uses, yielding a nontrivial accuracy–coverage–set-efficiency frontier.
5. Selecting adaptation examples before forming the calibration pool can bias calibration; calibration-first partition materially repairs this failure mode.
6. In the tested three-target study, the alpha-derived conservative n_cal=19 validity-first operating point achieved >=90% mean coverage in all 12 task x budget cells in the initial fold-averaged audit; publication-grade pooled metrics are being regenerated before final numerical claims.

## Claims not allowed
- A universal optimal adaptation/calibration fraction.
- A universal best acquisition score.
- A universal minimum of 19 calibration labels for all datasets, all alpha values, or conditional coverage.
- Distribution-free conditional coverage.
- Deployment readiness.
- General agricultural universality beyond the tested crops/datasets.
- Novelty of DINOv2, conformal prediction, active learning, or label-budget allocation as generic concepts.

## Remaining submission blockers
1. Regenerate exact pooled held-out publication metrics and uncertainty intervals for the main validity-first and lean operating points.
2. Freeze final contribution-to-result matrix from those pooled metrics.
3. Run a final literature/novelty stress test specifically against joint adaptation + conformal calibration budgeting under distribution shift.
4. Only after these gates, begin manuscript drafting and venue selection.

## Readiness status
**REVISE -> near writing gate.** Scientific identity is stable; remaining blockers are publication-grade metric regeneration, final claim audit, and novelty stress testing rather than exploratory method search.
