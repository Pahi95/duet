# DUET

**D**etection–**E**xpression **U**nified **T**est — a two-part hurdle model for
single-cell differential expression, in pure Python.

DUET is a from-scratch implementation of the MAST-style hurdle test: a logistic
**detection** model and a Gaussian **positive-expression** model, whose
likelihood-ratio statistics are summed into a single χ² test. Two voices, one
piece.

It works directly on sparse `AnnData`, with no R toolchain and no writing the
expression matrix to disk. On 11 106 cells it is **117× faster than R MAST at the
model fit and 224× end-to-end, in 5.2 GB less memory** — while reproducing MAST's
coefficients and gene ranking essentially exactly.

> Formerly published in this project as `scPyDE`. Renamed in July 2026 because
> that name read as "Python SCDE", and [SCDE](https://doi.org/10.1038/nmeth.2967)
> (Kharchenko et al. 2014) is an established but statistically unrelated method —
> a Bayesian dropout error model, not a hurdle.

---

## The model

For one gene in one cell type, with `y` the log-normalized expression and `cond`
the condition indicator:

**Part 1 — detection (logistic).**

```
y_detect = 1[y > 0]
full:  y_detect ~ 1 + cond + CDR (+ covariates)
null:  y_detect ~ 1        + CDR (+ covariates)
stat_detect = 2 * (loglik_full - loglik_null)
```

**Part 2 — positive expression (Gaussian OLS), on cells with `y > 0` only.**

```
stat_continuous = n_pos * ln(RSS_null / RSS_full)
```

**Combined.**

```
stat_hurdle = stat_detect + stat_continuous
df_hurdle   = df_detect   + df_continuous
pvalue      = chi2.sf(stat_hurdle, df_hurdle)
fdr         = Benjamini-Hochberg within each (comparison, cell type) block
```

CDR (cellular detection rate) is the default nuisance covariate. Empirical-Bayes
variance moderation of the continuous component is on by default under
`mast_compat`.

## Install

No package on PyPI yet. Clone and put the repository on your path:

```bash
git clone https://github.com/Pahi95/duet.git
cd duet
pip install numpy scipy pandas anndata scanpy      # tqdm, psutil optional
```

Tested with Python 3.12, NumPy 2.0, SciPy 1.14, AnnData 0.12.

## Quick start

```python
import anndata as ad
from duet import run_duet

adata = ad.read_h5ad("your_data.h5ad")     # X = log-normalized

out = run_duet(
    adata,
    output_dir="results",
    celltype_col="celltype",
    sample_col="Sample",       # the condition column
    ref_label="ctrl",
    test_label="stim",
    mast_compat=True,          # MAST's gating, partial hurdle, EB shrinkage
    vectorized=True,           # closed-form two-group fast path
)
# -> results/duet_all_celltypes.csv
```

`mast_compat=True` reproduces R MAST's pipeline gating: `min_cells_per_group=1`,
`min_total_cells=20`, `min_positive=2`, no covariates, partial hurdle, and EB
shrinkage. Drop it for the fuller model with the CDR covariate.

## Output

One row per gene per cell type:

| column | meaning |
|---|---|
| `gene`, `primerid` | gene id (`primerid` for MAST compatibility) |
| `coef` | hurdle log fold change (see *Fold-change scale*) |
| `pvalue`, `Pr(>Chisq)` | hurdle p-value |
| **`neglog10p`** | **underflow-safe −log10(p) — use this for ranking** |
| `fdr` | Benjamini–Hochberg, within cell type |
| `stat_detect`, `stat_continuous`, `stat_hurdle` | the two components and their sum |
| `p_detect`, `p_continuous` | per-component p-values |
| `detect_rate_ref/test`, `mean_expr_ref/test` | descriptive counts |
| `tested`, `skip_reason` | why a gene was or was not tested |

### Rank on `neglog10p`, not `pvalue`

`chi2.sf` returns exactly `0.0` once the hurdle statistic passes ~1400, which at
realistic cell counts hits **3.5–6.9 % of genes**. Those genes then tie at
`p == 0` even though their true −log10 p spans 311 → 3258, and any "top-K by
p-value" selection picks among them arbitrarily. Measured effect: **top-100 gene
overlaps shift by up to 46 genes**.

`neglog10p` is computed without passing through the float64 p-value, using the
asymptotic expansion of the χ² upper tail — exact for `df = 2` (the standard
hurdle), max absolute error `0.0` against `scipy.stats.chi2.logsf` wherever SciPy
can still compute it. `pvalue` keeps its underflowed zeros so the MAST-compatible
output contract is unchanged.

R MAST and diffxpy hit the same floor but cannot be repaired after the fact:
MAST's output carries no test statistic, and diffxpy's is a coefficient.

### Fold-change scale

`coef` defaults to the **compressed** MAST-style log-mean difference
(`mean_expr_test − mean_expr_ref`), which matches R MAST. Pass
`linear_logfc=True` for the scanpy/Seurat **linear** log2 fold change
(`log2((expm1(m_test)+ε)/(expm1(m_ref)+ε))`).

They are not interchangeable, and this trips people up: in linear mode DUET
computes scanpy's `logfoldchanges` *character for character*, so correlating the
two is a formula against itself, not a validation. Note also that the linear
scale gives |log2FC| ≈ 28 for genes detected in only one group — an artefact of
`ε`, inherited from scanpy's definition.

## Validation

Benchmarked against R MAST 1.33.0, diffxpy, PyDESeq2 and scanpy's Wilcoxon test on
three datasets — human pancreas (PDAC, 229k cells), [Kang et al. 2018](https://doi.org/10.1038/nbt.4042)
(PBMC + IFN-β, paired donors) and [Crowell et al. 2020](https://doi.org/10.1038/s41467-020-19894-4)
(mouse cortex, LPS 4v4) — plus a muscat ground-truth simulation.

**Agreement with R MAST**, identical cells, genes and design (`~1+condition`):

| dataset | log2FC Pearson | sign | −log10p Spearman | top-100 overlap |
|---|---|---|---|---|
| Pancreas | 1.000 | 100 % | 0.9999 | 100/100 |
| Kang 2018 | 1.000 | 100 % | 1.000 | 100/100 |
| Crowell 4v4 | 1.000 | 100 % | 0.997–1.000 | 98–100/100 |

**Speed** (Ductal cell, ~8.5k genes, same machine):

| cells | DUET | R MAST (fit) | R MAST (end-to-end) | MAST peak RAM |
|---:|---:|---:|---:|---:|
| 1 000 | 1.19 s | 28.1 s | 55.0 s | 1.2 GB |
| 11 106 | **1.96 s** | 227.9 s | 438.6 s | 5.2 GB |

Per-cell cost is 0.000072 s for DUET against 0.0188 s for MAST — a factor of 260,
so the margin widens with dataset size. DUET streams one sparse gene column at a
time; MAST densifies at ~0.39 MB per cell.

## Limitations — read before reporting gene counts

**DUET's FDR is not usable across donors.** On a muscat simulation with realistic
between-sample variance (4v4 samples, 405 true DE genes), at a nominal FDR of
0.05:

| method | AUC | power | **observed FDR** |
|---|---:|---:|---:|
| PyDESeq2 (pseudobulk) | 0.997 | 0.71 | **0.00** |
| diffxpy | 0.992 | 0.95 | 0.07 |
| **DUET** | 0.978 | 0.99 | **0.81** |
| R MAST | 0.978 | 0.99 | **0.80** |
| Wilcoxon | 0.976 | 0.78 | 0.12 |

This is not a bug and not specific to DUET — it is the pseudoreplication bias
described by [Squair et al. 2021](https://doi.org/10.1038/s41467-021-25960-2) and
[Zimmerman et al. 2021](https://doi.org/10.1038/s41467-021-21038-1). Thousands of
cells from a handful of donors are not thousands of independent observations, and
DUET reproduces MAST's behaviour here as faithfully as everywhere else. Under a
label-permutation null DUET is correctly calibrated (type-I error 0.034–0.059
against a nominal 0.05); it is the independence assumption that fails, not the
test.

How badly depends on the experiment. Splitting the reference samples of each
dataset at random and asking for differential expression between two groups that
differ by nothing:

| dataset | DUET | Wilcoxon | PyDESeq2 (pseudobulk) |
|---|---:|---:|---:|
| Pancreas (16 donors, separate matrices) | 75–99 % | 33–53 % | **0.00–0.85 %** |
| Kang 2018 (8 donors, multiplexed pool) | 1–27 % | 0.6–10 % | **0.00 %** |
| Crowell (4 mice, separate preps) | ~100 % | 6–89 % | **0.00–0.37 %** |

The penalty tracks between-sample *batch* variance, not donor count — Kang has
half of pancreas's donors and does far better because all its donors were pooled
into shared 10x runs and demultiplexed genetically.

**In practice: rank with DUET, call with pseudobulk.**

### Where the hurdle earns its keep

The flip side, from the same simulation — % of true positives detected by category:

| true change | n | DUET | PyDESeq2 | Wilcoxon |
|---|---:|---:|---:|---:|
| mean shift | 113 | **100.0** | 93.8 | 92.9 |
| differential proportion | 104 | **100.0** | 77.9 | 82.7 |
| differential modality | 104 | **98.1** | 73.1 | 78.8 |
| **bimodal / both** | 83 | **98.8** | **28.9** | 53.0 |

Where the change is in the *detection rate* rather than the mean, a mean-based
pseudobulk test is structurally blind and the hurdle is 3.4× better. That is what
the detection component is for.

## Known issues

- **The EB statistic saturates at 1373.87.** The empirical-Bayes step clips its
  moderated p-value at `1e-300` before inverting back to a χ²₁ statistic, so the
  continuous component cannot exceed `chi2.isf(1e-300, 1)`. This affects ~1 % of
  genes, all astronomically significant either way, and leaves ranking untouched
  (Spearman 1.0000 against MAST). Fixable with the same log-space technique as
  `neglog10p`, but it would change p-values and FDR, so it is not applied.
- Under a permutation null on one cell type (pancreas Ductal) the type-I error is
  0.203 rather than ~0.05; the other eight dataset × cell-type combinations are
  0.034–0.059. Unexplained.

## Compatibility

`run_scpyde` and `run_python_hurdle_de` remain as aliases for `run_duet`.

## References

- Finak G, McDavid A, Yajima M, *et al.* MAST: a flexible statistical framework for assessing transcriptional changes and characterizing heterogeneity in single-cell RNA sequencing data. *Genome Biology* 16:278 (2015). [doi:10.1186/s13059-015-0844-5](https://doi.org/10.1186/s13059-015-0844-5)
- Squair JW, Gautier M, Kathe C, *et al.* Confronting false discoveries in single-cell differential expression. *Nature Communications* 12:5692 (2021). [doi:10.1038/s41467-021-25960-2](https://doi.org/10.1038/s41467-021-25960-2)
- Zimmerman KD, Espeland MA, Langefeld CD. A practical solution to pseudoreplication bias in single-cell studies. *Nature Communications* 12:738 (2021). [doi:10.1038/s41467-021-21038-1](https://doi.org/10.1038/s41467-021-21038-1)
- Crowell HL, Soneson C, Germain P-L, *et al.* muscat detects subpopulation-specific state transitions from multi-sample multi-condition single-cell transcriptomics data. *Nature Communications* 11:6077 (2020). [doi:10.1038/s41467-020-19894-4](https://doi.org/10.1038/s41467-020-19894-4)
- Kang HM, Subramaniam M, Targ S, *et al.* Multiplexed droplet single-cell RNA-sequencing using natural genetic variation. *Nature Biotechnology* 36:89–94 (2018). [doi:10.1038/nbt.4042](https://doi.org/10.1038/nbt.4042)
- Smyth GK. Linear models and empirical Bayes methods for assessing differential expression in microarray experiments. *SAGMB* 3:Article 3 (2004). [doi:10.2202/1544-6115.1027](https://doi.org/10.2202/1544-6115.1027)

## License

Not yet licensed. Until a license is added this is, legally, all rights reserved.

## Status

Research code under active development. The benchmarking and validation harness
that produced the numbers above lives in a separate analysis project and is not
part of this repository.
