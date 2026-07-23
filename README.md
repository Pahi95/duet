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
coefficients and gene ranking essentially exactly. The margin grows with population
size, reaching **149× at the fit and 282× end-to-end at 23 895 cells**, because
MAST is linear in cells where DUET is nearly flat.

> **The speed figures are for the two-group design without covariates**, where DUET
> takes a fully vectorised closed-form path. Add a covariate and there is no closed
> form, so it fits each gene iteratively: DUET takes 78 s where MAST's fit and test
> take 57 s, i.e. **MAST is ~1.4× faster on that path**. Accuracy is unaffected —
> the covariate path reproduces MAST just as closely (log2FC *r* = 0.9994,
> −log10 p Spearman 0.9998).

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

```bash
git clone https://github.com/Pahi95/duet.git
cd duet
pip install -e .
```

Requires **Python ≥ 3.11** — that floor comes from `anndata`, not from DUET,
whose engine imports only NumPy, SciPy and pandas. Optional extras:

```bash
pip install -e ".[extra]"   # statsmodels (alternative logistic engine), tqdm, psutil
pip install -e ".[test]"    # pytest, scikit-learn
```

All three optional packages degrade gracefully if absent. Results in this README
were produced on Python 3.12, NumPy 2.0, SciPy 1.14, AnnData 0.12; CI runs the
test suite on 3.11 and 3.12. Not yet on PyPI.

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
| `detection_separated` | a group detects the gene in exactly 0 % or 100 % of cells — the test is valid, the **coefficient is not** |
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
three datasets — an integrated human pancreas dataset (PDAC, unpublished), [Kang et al. 2018](https://doi.org/10.1038/nbt.4042)
(PBMC + IFN-β, paired donors) and [Crowell et al. 2020](https://doi.org/10.1038/s41467-020-19894-4)
(mouse cortex, LPS 4v4) — plus a muscat ground-truth simulation.

**Agreement with R MAST**, identical cells, genes and design (`~1+condition`):

| dataset | model | log2FC Pearson | sign | −log10p Spearman | top-100 overlap |
|---|---|---|---|---|---|
| Pancreas | `~1+condition` | 1.000 | 100 % | 0.9999 | 100/100 |
| Kang 2018 | `~1+condition` | 1.000 | 100 % | 1.000 | 100/100 |
| Crowell 4v4 | `~1+condition` | 1.000 | 100 % | 0.997–1.000 | 98–100/100 |
| Kang 2018 | **`~1+CDR+condition`** | 0.9994 | 98.9 % | 0.9998 | 100/100 |

The last row exercises the general per-gene path (the vectorised one applies only
without covariates), so both code paths are validated against MAST.

**Speed**, ~8 000 genes throughout, same machine. The Ductal series is the largest
single cell type with a balanced two-group split; the pooled series (Ductal +
Endothelial + Stellate) carries the same measurement further. Both are cost
benchmarks — the pooled arms differ in composition, so no agreement statistic is
taken from them.

| population | cells | DUET | R MAST (fit) | R MAST (e2e) | speed-up (e2e) | MAST peak RAM |
|---|---:|---:|---:|---:|---:|---:|
| Ductal | 1 000 | 1.19 s | 28.1 s | 55.0 s | 46× | 1.2 GB |
| Ductal | 11 106 | **1.96 s** | 227.9 s | 438.6 s | 224× | 5.2 GB |
| pooled | 2 500 | 1.80 s | 54.4 s | 101.2 s | 56× | 1.7 GB |
| pooled | 10 000 | 2.34 s | 215.3 s | 409.4 s | 175× | 4.4 GB |
| pooled | **23 895** | **3.36 s** | 499.9 s | 947.9 s | **282×** | 9.4 GB |

Per-cell cost is 0.000072 s for DUET against ~0.02 s for MAST, reproduced
independently on both populations. MAST's peak memory is linear at **0.368 MB per
cell (r² = 1.0000)**; DUET streams one sparse gene column at a time, so its
footprint stays flat. Note that cell-level testing runs within a cell type, so the
population size — not the size of the whole object — is what sets the cost.

## Limitations — read before reporting gene counts

**DUET's FDR is not usable across donors.** On a muscat simulation with realistic
between-sample variance (4v4 samples, 405 true DE genes), at a nominal FDR of
0.05:

Averaged over **five independent simulation replicates** (a single draw is not
enough — see the range column):

| method | AUC | power | **observed FDR** | FDR range |
|---|---:|---:|---:|---:|
| PyDESeq2 (pseudobulk) | 0.984 ± 0.008 | 0.67 ± 0.04 | **0.000** | 0.000 – 0.000 |
| **DUET** | 0.981 ± 0.022 | 0.97 ± 0.02 | **0.39 ± 0.38** | **0.047 – 0.873** |
| R MAST | 0.981 ± 0.022 | 0.97 ± 0.02 | 0.39 ± 0.38 | 0.045 – 0.873 |
| Wilcoxon | 0.973 ± 0.007 | 0.76 ± 0.09 | 0.08 ± 0.16 | 0.003 – 0.360 |

**The spread is the result.** DUET controlled FDR in two replicates of five (0.047,
0.072) and failed badly in the others (0.26, 0.70, 0.87) — on data generated the
same way each time. Cell-level FDR is therefore not uniformly inflated but
*unpredictable*, which is worse for a practitioner: you cannot tell from one run
whether your calls are trustworthy. Pseudobulk returned exactly 0.000 every time.
DUET and MAST tracked each other to within 0.012 in every replicate.

This is not a bug and not specific to DUET — it is the pseudoreplication bias
described by [Squair et al. 2021](https://doi.org/10.1038/s41467-021-25960-2) and
[Zimmerman et al. 2021](https://doi.org/10.1038/s41467-021-21038-1). Thousands of
cells from a handful of donors are not thousands of independent observations, and
DUET reproduces MAST's behaviour here as faithfully as everywhere else. Under a
label-permutation null DUET is correctly calibrated (type-I error 0.034–0.059
across nine dataset × cell-type combinations, nominal 0.05); it is the
independence assumption that fails, not the test.

A note on how much a single permutation tells you: on pancreas Ductal cells one
draw (seed 0) gave 0.203. Repeating over ten seeds gives 0.034–0.075 for the other
nine, mean **0.047** excluding that draw and 0.063 including it, with **zero**
genes surviving FDR correction in 9/10 seeds. A single permutation is a single
draw from a null distribution that has real variance, because the cells are not
exchangeable across donors — quote a mean over seeds, not one run. The residual
inflation that does exist sits in sparsely detected genes (type-I 0.098 at
detection rate ~0.18, falling to 0.036 at ~0.68), which is what
`--duet-min-detect-frac` is for.

How badly depends on the experiment. Splitting the reference samples of each
dataset at random and asking for differential expression between two groups that
differ by nothing:

Repeated over **ten independent random splits per cell type** (90 splits total),
because a single split turns out to be uninformative — the spread is enormous:

| dataset | DUET mean ± sd | range | Wilcoxon | PyDESeq2 (pseudobulk) |
|---|---:|---:|---:|---:|
| Pancreas (16 donors, separate matrices) | 74.6 ± 33.6 % | 1.0–100 % | 30.5 % | **0.13 %** |
| Kang 2018 (8 donors, multiplexed pool) | **24.6 ± 25.8 %** | 0.9–94.2 % | 7.5 % | **0.02 %** |
| Crowell (4 mice, separate preps) | 83.4 ± 22.8 % | 33.9–100 % | 37.9 % | **0.09 %** |

**Pseudobulk never exceeded 5 % in any of the 90 splits** (mean 0.08 %, max 2.47 %).

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
  continuous component cannot exceed `chi2.isf(1e-300, 1)`.

  Measured impact on 11 106 cells: **101 of 8 497 genes (1.19 %)** sit at the cap.
  Within that set DUET's statistic compresses to 1315–1433 where MAST spans
  1311–8935, so their relative ordering degrades (Spearman 0.894 within the capped
  set). Globally the effect is invisible — statistic Spearman against MAST is
  **0.9999** — and **no capped gene changes significance call** (all remain
  FDR < 0.05). Practical consequence: if you rank the ~100 most extreme genes by
  statistic, that ordering is partly arbitrary; every threshold-based result is
  unaffected.

  Not fixed, and not fixable by the `neglog10p` route: that works because the χ²
  tail has a closed-form asymptotic, whereas here the input is a moderated
  *t* statistic and SciPy's `t.logsf` itself underflows to `-inf` above
  |t| ≈ 40 at these degrees of freedom. A correct fix needs a log-space *t* tail,
  which would change p-values and FDR.

## Tests

```bash
pytest tests/ -q
```

18 tests covering the underflow-safe tail against SciPy, BH correctness, the
logistic engine (including separation), and end-to-end invariants — that the
vectorised and general code paths give the same answer, that a pure
detection-rate change is detected, and that null data stays calibrated. Run on
Python 3.10 and 3.12 by GitHub Actions.

## Compatibility

`run_scpyde` and `run_python_hurdle_de` remain as aliases for `run_duet`.

**`logistic_engine` now defaults to `"numpy_irls"`** (it was `"statsmodels"`). This
affects only runs *with covariates* — the vectorised two-group path fits no GLM at
all. It is 2.8–4.4× faster for the same fit, and agreement is exact on 80,947 of
80,950 gene-by-cell-type tests measured across three datasets. The three exceptions
are genes detected in exactly 100% of one group, where the maximum-likelihood
estimate does not exist and no solver is correct. Pass
`logistic_engine="statsmodels"` to reproduce results generated before this change.

## References

- Finak G, McDavid A, Yajima M, *et al.* MAST: a flexible statistical framework for assessing transcriptional changes and characterizing heterogeneity in single-cell RNA sequencing data. *Genome Biology* 16:278 (2015). [doi:10.1186/s13059-015-0844-5](https://doi.org/10.1186/s13059-015-0844-5)
- Squair JW, Gautier M, Kathe C, *et al.* Confronting false discoveries in single-cell differential expression. *Nature Communications* 12:5692 (2021). [doi:10.1038/s41467-021-25960-2](https://doi.org/10.1038/s41467-021-25960-2)
- Zimmerman KD, Espeland MA, Langefeld CD. A practical solution to pseudoreplication bias in single-cell studies. *Nature Communications* 12:738 (2021). [doi:10.1038/s41467-021-21038-1](https://doi.org/10.1038/s41467-021-21038-1)
- Crowell HL, Soneson C, Germain P-L, *et al.* muscat detects subpopulation-specific state transitions from multi-sample multi-condition single-cell transcriptomics data. *Nature Communications* 11:6077 (2020). [doi:10.1038/s41467-020-19894-4](https://doi.org/10.1038/s41467-020-19894-4)
- Kang HM, Subramaniam M, Targ S, *et al.* Multiplexed droplet single-cell RNA-sequencing using natural genetic variation. *Nature Biotechnology* 36:89–94 (2018). [doi:10.1038/nbt.4042](https://doi.org/10.1038/nbt.4042)
- Smyth GK. Linear models and empirical Bayes methods for assessing differential expression in microarray experiments. *SAGMB* 3:Article 3 (2004). [doi:10.2202/1544-6115.1027](https://doi.org/10.2202/1544-6115.1027)

## License

MIT — see [LICENSE](LICENSE).

## Status

Research code under active development. The benchmarking and validation harness
that produced the numbers above lives in a separate analysis project and is not
part of this repository.
