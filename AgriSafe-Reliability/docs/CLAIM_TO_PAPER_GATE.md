# AgriSafe Claim-to-Paper Gate — Publication-Grade Pooled Metrics

## Gate outcome
**READY WITH WARNINGS for manuscript drafting.**

The scientific identity is now stable enough to write. The remaining work is manuscript construction, a final focused novelty comparison, figure/table generation, and reviewer-style stress testing. Additional exploratory method search is not currently justified.

## C1 — Field-label budgeting is a joint prediction-and-validity problem
**Claim.** Under lab-to-field shift, a fixed target-label budget has two competing uses: predictor adaptation and target-domain calibration. Accuracy-only and calibration-only allocations occupy different ends of a nontrivial accuracy–coverage–set-efficiency frontier.

**Evidence.** Replicated on three field targets:
1. PlantVillage corn -> PlantDoc corn.
2. PlantVillage corn -> primary-clean PlantWild corn.
3. PlantVillage apple -> PlantDoc apple.

The Apple task changes crop and is free of exact target duplicate conflicts, reducing the likelihood that the observed frontier is a corn-only or PlantWild-cleaning artifact.

**Status: VALIDATED for the tested tasks and protocol.**

## C2 — Acquisition signals themselves are shift-sensitive
**Claim.** A fixed target-acquisition score does not transfer uniformly across field domains or label budgets.

**Evidence.**
- Corn PlantDoc: entropy is strong at very low budget; committee predictive entropy can be severely harmful.
- Corn PlantWild: committee disagreement helps at B=3/6, while single entropy becomes competitive later.
- Apple PlantDoc: entropy is worse than random at low budgets; margin or least-confidence is stronger at several budgets.

**Status: VALIDATED as a bounded empirical finding.**

**Boundary.** This does not prove that no universal acquisition method can exist. It establishes only that the tested standard uncertainty/diversity/committee signals are not distribution-shift invariant across these targets.

## C3 — Calibration-first validity budgeting
**Claim.** When adaptation examples are actively selected, target calibration examples should be randomly reserved before active acquisition so that the calibration pool is not depleted of difficult examples. A conservative alpha-derived calibration floor provides a validity-first operating point.

### Mechanism evidence
A post-hoc proportional reserve failed strict leave-one-task-out transfer. Diagnosing the failure showed that the earlier protocol selected adaptation examples first and calibrated from the remaining pool, allowing active selection to distort the calibration sample. Calibration-first partition materially repaired the problematic cells.

### Alpha-derived conservative floor
At alpha=0.1, n_cal=19 is the smallest calibration size for which the standard split-conformal order statistic k=ceil((n+1)(1-alpha)) is strictly below n rather than being forced to the maximum calibration nonconformity score.

### Publication-grade pooled metrics
Metrics were recomputed by pooling all held-out fold predictions/sets before calculating global accuracy, macro-F1, coverage, and set efficiency. Thirty random calibration-first repeats were used for each task x budget x operating point.

For the conservative n_cal=19 operating point, all 12 task x budget cells achieved pooled mean empirical coverage above 0.90:
- Corn -> PlantDoc: 0.9520, 0.9465, 0.9491, 0.9535 for B=20,40,80,160.
- Corn -> PlantWild: 0.9533, 0.9509, 0.9472, 0.9532.
- Apple -> PlantDoc: 0.9552, 0.9526, 0.9661, 0.9573.

Minimum pooled mean coverage: **0.9465**.

The lean n_cal=10 regime retains more labels for adaptation but does not uniformly preserve the nominal level: Apple B=160 has pooled mean coverage **0.8988**.

**Status: VALIDATED WITH BOUNDARY.**

**Boundary.** n_cal=19 is a conservative validity-first operating point, not a universal optimum and not a guarantee of conditional coverage. Its conformal interpretation requires exchangeability between the randomly reserved target calibration examples and deployment/test examples, and the calibration examples must remain outside active adaptation selection and training.

## C4 — Coverage recovery is not equivalent to useful reliability recovery
**Claim.** Target calibration can restore nominal marginal coverage while remaining operationally weak when prediction sets become broad.

**Evidence.**
- Corn targets require substantially broader sets under conservative validity, often above 2 classes on average in a 3-class task.
- Apple target achieves comparable coverage with materially tighter sets, showing that the operational cost of validity is domain dependent.
- At low total budgets, reserving 19 labels for validity leaves almost no adaptation capacity, exposing a real feasibility boundary rather than a free improvement.

**Status: VALIDATED for the tested tasks.**

## Main table anchors from pooled results
### Corn -> PlantDoc
- B=20, cal=10: accuracy 0.6488, coverage 0.9053, set size 2.044.
- B=20, cal=19: accuracy 0.5336, coverage 0.9520, set size 2.629.
- B=160, cal=10: accuracy 0.7550, coverage 0.9138, set size 1.818.
- B=160, cal=19: accuracy 0.7547, coverage 0.9535, set size 2.093.

### Corn -> PlantWild
- B=20, cal=10: accuracy 0.5304, coverage 0.9112, set size 2.451.
- B=20, cal=19: accuracy 0.4925, coverage 0.9533, set size 2.725.
- B=160, cal=10: accuracy 0.7160, coverage 0.9208, set size 2.054.
- B=160, cal=19: accuracy 0.6987, coverage 0.9532, set size 2.340.

### Apple -> PlantDoc
- B=20, cal=10: accuracy 0.6512, coverage 0.9216, set size 1.910.
- B=20, cal=19: accuracy 0.6416, coverage 0.9552, set size 2.055.
- B=160, cal=10: accuracy 0.8287, coverage 0.8988, set size 1.329.
- B=160, cal=19: accuracy 0.8259, coverage 0.9573, set size 1.615.

## Evidence artifact
Primary publication-QA result:
`journal_extension/results/phase11_publication_pooled_metrics.json`

## Claims explicitly prohibited
- Universal optimal adaptation/calibration ratio.
- Universal best acquisition score.
- Universal sufficiency or optimality of 19 calibration labels.
- Conditional coverage guarantees.
- Deployment readiness.
- General agricultural universality beyond the tested settings.
- Novelty claims for DINOv2, active learning, conformal prediction, or label-budget allocation in isolation.

## Remaining work before submission
1. Freeze final paper title/contribution wording after the focused novelty comparison.
2. Generate publication tables and figures directly from `phase11_publication_pooled_metrics.json` and acquisition result artifacts.
3. Draft the manuscript around C1-C4 without adding unsupported method claims.
4. Run a fresh reviewer-style stress test before selecting/finalizing the venue.
