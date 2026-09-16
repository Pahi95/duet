# DUET

**Detection–Expression Unified Test** provides a cell-level two-part hurdle test
and a donor-level pseudobulk arm for single-cell RNA sequencing in Python.

The detection component uses logistic regression. The positive-expression
component uses a Gaussian model; when empirical Bayes is enabled, its moderated
t tail is converted to a χ² statistic before combination with detection. This
continuous statistic is not numerically identical to MAST's moderation.

## Install

```sh
pip install -e ".[pseudobulk]"
```

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

The `scripts/` directory contains benchmarking, simulation and numerical
validation utilities. These require separately prepared input data and the
appropriate R/Python dependencies. Generated results, manuscript files and
figures are kept outside the versioned repository.

## Tests and license

```sh
pip install -e ".[test]"
pytest
```

MIT; see [LICENSE](LICENSE). Citation metadata: [CITATION.cff](CITATION.cff).
