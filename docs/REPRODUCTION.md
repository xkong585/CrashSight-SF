# Local reproduction record

The portable script was executed on the bundled raw ZIP, with 61,702 records and two duplicate case IDs removed. Recomputed charts use navy/teal/gold and a light background.

## Changes from the original workflow

- Replaced Colab mounts and machine-specific working directories with repository-relative paths.
- Preserved missing context using explicit categories rather than assuming missing time means midnight.
- Flagged the partial final year in the main chart.
- Added a distinct severity baseline with training through 2022, threshold selection on 2023, and a 2024 test. This is not directly comparable to the original random-split grid-risk AUC.
- Anchored the siting window to the latest dataset date instead of the current system date.
- Used a deterministic greedy location heuristic with explicit distance and candidate assumptions. No OPTIMAL solver status is asserted.
- Regenerated all figures featured in the new README from computed arrays. Original image files remain historical outputs.

Original Colab notebooks, their outputs, global/local spatial autocorrelation results, and patrol-routing results are retained but were not executed end-to-end in this refresh. The new notebook and script are the supported portable reproduction path. Road-network response time and policy benefit remain outside scope.
