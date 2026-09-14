# AgriSafe PRL reproducibility bundle

## Scope
This bundle supports the manuscript **Validity-First Field-Label Budgeting for Reliable Plant Disease Recognition under Distribution Shift**. It contains frozen protocol documents, target/source split manifests, experiment scripts, publication result JSON files, and figure/table source data. Raw public datasets and cached DINOv2 feature arrays are intentionally not redistributed.

## Public datasets
- PlantVillage: curated source images.
- PlantDoc: real-field target images for corn and apple.
- PlantWild: real-field target images for corn. The primary analysis uses the audited `primary_clean` subset described in the integrity documents and manifest.

Obtain each dataset from its original provider and preserve the sample identities in the supplied manifests.

## Frozen representation and source model
- Frozen DINOv2-small CLS features.
- L2-normalized features.
- Multinomial logistic-regression linear probes.
- Source regularization was selected from source validation only and frozen before target experiments.
- Target held-out labels are evaluation-only.

## Publication protocol
1. Create five deterministic target folds from path hashes.
2. Hold one target fold out for evaluation.
3. For joint budgeting, randomly reserve calibration examples **before** active adaptation selection.
4. Restrict active adaptation selection to the remaining candidate pool.
5. Fit the adapted source+target linear probe.
6. Calibrate split-conformal prediction sets on the reserved target calibration sample.
7. Evaluate on the untouched held-out fold.
8. Concatenate held-out predictions across the five folds within each repeat and compute global publication metrics.
9. Repeat 30 times per target-budget-operating-point cell for the final pooled analyses.

## Key publication artifacts
- `phase11_publication_pooled_metrics.json`: lean (`n_cal=10`) and validity-first (`n_cal=19`) pooled operating points.
- `phase12_publication_full_allocation_grid.json`: full adaptation/calibration allocation grid.
- `fig*_source.csv`: exact numerical sources behind publication figures.
- `Table*_*.csv`: publication tables.

## Scripts
- `publication_pooled_metrics.py`: final pooled lean/validity-first recomputation.
- `publication_full_allocation_grid.py`: final pooled full allocation grid.
- `make_publication_fig1_fig3_fig5.py`, `make_publication_fig2_and_tables.py`, `make_publication_fig4_and_table3.py`: deterministic figure/table generation.
- Phase scripts document the active acquisition, target calibration, third-domain, and validity-floor analyses that precede the final pooled publication pass.

## Integrity boundaries
- Do not substitute exploratory fold-average numbers for pooled publication metrics.
- Do not use held-out target labels for acquisition scoring, hyperparameter selection, fitting, or calibration.
- Do not restore PlantWild conflict/duplicate files removed by the frozen primary-clean audit.
- Do not interpret `n_cal=19` as a universal optimum or minimum; it is a conservative alpha=0.1 operating point evaluated in this study.
- Do not claim conditional conformal coverage or deployment readiness.

## Environment
The project was executed on Linux with Python, NumPy, scikit-learn, PyTorch, Transformers, and Matplotlib. DINOv2 feature extraction used the project's existing `llama_factory` environment (recorded during extraction as PyTorch 2.7.1+cu126, Transformers 4.52.1, scikit-learn 1.7.0). The frozen result files are the authoritative source for manuscript-number verification.

## Manuscript-number provenance
The submission PDFs should be checked against the JSON/CSV artifacts in this bundle. The bundle supports the submitted analysis but does not redistribute third-party image datasets.
