# DUET

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22801244.svg)](https://doi.org/10.5281/zenodo.22801244)
[![PyPI](https://img.shields.io/pypi/v/duet-de.svg)](https://pypi.org/project/duet-de/)

**Detection–Expression Unified Test** provides a cell-level two-part hurdle test
and a donor-level pseudobulk arm for single-cell RNA sequencing in Python.

The detection component uses logistic regression. The positive-expression
component uses a Gaussian model; when empirical Bayes is enabled, its moderated
t tail is converted to a χ² statistic before combination with detection. This
continuous statistic is not numerically identical to MAST's moderation.

## Install

```sh
pip install "duet-de[pseudobulk]"
```

From a clone of this repository: `pip install -e ".[pseudobulk]"`.

Python 3.11 or later is supported. The core uses NumPy, SciPy and pandas with
AnnData input. PyDESeq2 0.5 or later is required for the pseudobulk arm.
MAST/R is needed only for comparison analyses, not for DUET itself.

## Cell-level scores

```python
import pandas as pd
from duet import run_duet

# adata.X contains log1p-normalized expression; obs columns identify cell type
# and condition. sample_col names the CONDITION, not the donor identifier.
path = run_duet(adata, output_dir="results", celltype_col="celltype",
                sample_col="condition", ref_label="ctrl", test_label="stim",
                mast_compat=True, vectorized=True)
scores = pd.read_csv(path)
ranked = scores.sort_values("neglog10p", ascending=False)
```

Output rows retain input order. Sort `neglog10p` explicitly; ordinary p-values
may underflow to zero. Across multiple donors, cell-level significance can be
severely anti-conservative. These scores are exploratory, not donor-level tests.

## Donor-level calls and complementary scores

```python
from duet import run_duet_pseudobulk

# adata.layers["counts"] must contain raw integer counts with valid sample
# provenance; normalized or integrated expression is not a raw-count layer.
path = run_duet_pseudobulk(
    adata, output_dir="results", celltype_col="celltype",
    condition_col="condition", donor_col="donor",
    ref_label="ctrl", test_label="stim", counts_layer="counts",
    paired=False, n_cpus=1,
)
```

Set `paired=True` explicitly when every donor contributes both conditions.
The default `significant` call requires pseudobulk adjusted p < 0.05 and matching
effect directions. Other labels include `direction_conflict`, `ranked_only`,
`ns` and `not_tested`. A direction filter does not by itself prove FDR control.

`linear_logfc` computes a log2 ratio of **back-transformed group log1p means**,
not the ratio of arithmetic mean counts. Effect-size definitions differ across
methods, so thresholds cannot be transferred without checking the scale.

## Analysis and validation scripts

`scripts/` contains the analyses behind the DUET manuscript. They need the public input data
(Kang et al. 2018 PBMCs, Crowell et al. 2020 mouse cortex via muscatData, and the pancreas input
available from Zenodo as the analysis input: https://doi.org/10.5281/zenodo.22801671, to be placed at
`inputs/PancreasCorrected.h5ad`) and, for the comparisons, R with MAST, DESeq2, muscat, edgeR, limma,
lme4 and GLIMES. Results are written to `results/`, which is not versioned.
`python scripts/make_all_results.py --dry-run` lists every step and its command.

| Analysis | Scripts |
|---|---|
| Input data | `build_kang_h5ad.py`, `build_crowell_h5ad.py` (+ `get_crowell.R`), `build_pancreas_corrected.py` |
| Agreement with R MAST | `run_de_comparison.py`, `run_mast_dataset.py`, `mast_settings_check.py` (+ `mast_components.R`), `validate_covariate.py` (+ `.R`), `validate_engine.py` |
| Benchmark | `bench_prepare.py`, `bench_harness.py`, `bench_duet_run.py`, `bench_duet_run_matched.py`, `bench_mast_steps.R`, `collect_evidence.py` |
| p-value underflow | `underflow_check.py`, `eb_logspace_report.py` |
| muscat simulations (100 replicates) | `simulate_replicates.R`, `sim_calls.py` (+ `bench_mast_run.R`), `sim_score.py`, `sim_pb_edger_voom.R`, `sim_score_pb_extra.py` |
| Calibration diagnostics | `fdr_scale_diag.py`, `arm_offset.py`, `null_calibration.py`, `null_permutations.py`, `simulate_batch_nb.py` |
| Donor-swap null | `donor_swap_partitions.py`, `donor_swap_sexgenes.py` |
| Donor-aware models | `donor_aware_methods.py`, `donor_aware_run.R`, `install_glimes.R`, `evaluate_sim_replicates.py`, `evaluate_sim.py` |
| Pseudobulk implementation | `validate_pseudobulk_engines.py` (+ `pseudobulk_deseq2.R`) |

## Tests and license

```sh
pip install -e ".[test]"
pytest
```

MIT; see [LICENSE](LICENSE). Citation metadata: [CITATION.cff](CITATION.cff).
