# AgriSafe-Reliability

Code, frozen result artifacts, and source data for the manuscript:

**Validity-First Allocation of Field Labels for Reliable Plant Disease Recognition under Distribution Shift**
David Weng, Na Yang, and Yanhui Guo — submitted to *Pattern Recognition Letters*.

## What this is

Scarce field labels for plant-disease recognition under distribution shift must serve two
competing roles: adapting the classifier and calibrating its uncertainty. This repository
contains the complete analysis chain behind the paper's validity-first allocation study:
a calibration-first partition reserves a random target calibration sample *before* active
adaptation selection, and a conservative 19-label reserve is evaluated against a lean
10-label reserve under a per-fold label budget.

## Layout

- `scripts/` — feature extraction, acquisition, conformal recovery, joint allocation,
  leave-one-task-out audit, calibration-first partition, publication metrics, and
  figure/table generation (run order below).
- `results/` — frozen publication result JSON files (phases 1–12) that every number in
  the paper is generated from.
- `tables/` — figure source-data CSVs and Tables 1–3 source data.
- `figures/` — the five publication figures (PDF).
- `manifests/` — source split files and data-integrity manifests, including
  `plantdoc_corn_manifest.csv` (see the exclusion note below).
- `configs/` — frozen source-to-target class mappings.
- `docs/` — frozen protocol/audit documents.
- `REPRODUCIBILITY_README.md` — protocol description and regeneration notes.
- `environment.txt` — software versions used for the 2026-10-21 publication runs.

## Run order

1. Obtain PlantVillage, PlantDoc, and PlantWild from their original providers
   (raw datasets and cached DINOv2 feature arrays are not redistributed).
2. `dinov2_extract.py`, `extract_plantwild_corn_dinov2.py`, `extract_apple_dinov2.py`
   (feature extraction), then `build_plantdoc_corn_manifest.py`.
3. `active_label_budget_parallel.py`, `active_label_budget_plantwild.py` (phases 1–2);
   `qbc_acquisition_compare.py --target plantdoc|plantwild` (phase 3);
   `conformal_recovery_curve.py --target plantdoc|plantwild` (phase 4);
   `joint_budget_allocation.py --target plantdoc|plantwild` (phase 5);
   `apple_phase6_fast.py` (phase 6); `car10_budget_rule_fast.py` (phase 7);
   `car_loto_sensitivity.py` (phase 8); `calibration_first_partition.py` (phase 9);
   `calibration_first_validity_floor.py` (phase 10);
   `publication_pooled_metrics.py` (phase 11);
   `publication_full_allocation_grid.py` (phase 12).
4. `make_publication_fig2_and_tables.py`, `make_publication_fig4_and_table3.py`,
   `make_publication_fig1_fig3_fig5.py`.

## Data-integrity note (2026-10-21)

An exact-file (SHA-256) audit of the PlantDoc corn target found three byte-identical
image pairs carrying conflicting labels (each file appears under two classes in the
official train/test folders). Both copies of each pair were excluded as unknowable
(`primary_clean = 0` in `plantdoc_corn_manifest.csv`), leaving 370 of 376 images.
All publication analyses were regenerated after this exclusion. Label budgets are
per cross-fitting fold: within each repeat, every held-out fold's model spends its
own budget drawn from the remaining four folds.
