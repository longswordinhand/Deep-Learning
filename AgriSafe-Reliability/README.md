# AgriSafe Reliability: Reproducibility Materials

This directory accompanies the manuscript **“Validity-First Field-Label Budgeting for Reliable Plant Disease Recognition under Distribution Shift”** by Yanhui Guo, David Weng, and Na Yang.

The public release supports the manuscript's main numerical claims on lab-to-field transfer failure, acquisition instability, conformal recovery, and validity-first allocation under a scarce field-label budget.

## Public contents

- `REPRODUCIBILITY_README.md`: frozen evaluation protocol and integrity boundaries.
- `configs/corn_mapping.json`: audited corn label-space mapping.
- `scripts/publication_pooled_metrics.py`: final pooled lean/validity-first operating-point computation.
- `scripts/publication_full_allocation_grid.py`: final pooled adaptation/calibration allocation grid.
- `tables/fig1_transfer_failure_source.csv`: numerical source for Fig. 1.
- `tables/fig2_acquisition_delta_source.csv`: numerical source for Fig. 2.
- `tables/fig3_conformal_recovery_source.csv`: numerical source for Fig. 3.
- `tables/fig5_validity_budget_source.csv`: numerical source for Fig. 5 / validity-budget comparison.
- `tables/Table1_data_integrity.csv`: dataset and duplicate/conflict audit summary.

The public image datasets and cached DINOv2 feature arrays are **not redistributed** because they originate from third-party datasets/models. They should be obtained from their original providers. Additional frozen manifests and detailed audit artifacts are retained with the submission reproducibility bundle and can be released separately if required by the journal.

## Reproducibility boundary

The publication scripts use five path-hash target folds, calibration-first reservation, held-out target evaluation, and pooled metrics across held-out folds. Target evaluation labels must not be used for acquisition scoring, source-model selection, or other forbidden tuning steps. Negative findings are retained rather than rewritten as positive results.

The `n_cal=19` operating point is an alpha=0.1 conservative validity-first setting evaluated in this study; it is **not** claimed as a universal optimum or minimum, and the study does not claim conditional conformal coverage or deployment readiness.

## Authors

- Yanhui Guo — University of Illinois Springfield
- David Weng — Southern Illinois University Carbondale
- Na Yang — University of Illinois Springfield

Correspondence: Yanhui Guo, yguo56@uis.edu
