"""
core.py  --  DUET: the Detection-Expression Unified Test
=======================================================

A MAST-like two-part *hurdle* differential-expression engine implemented in
pure Python, designed as a drop-in alternative to the R/MAST backend used by
``scPipeline.py`` (selected there via ``de_method=DUET``).

The name is the model: a logistic **Detection** component and a Gaussian
positive-**Expression** component, sung as one piece -- their likelihood-ratio
statistics are summed into a single **Unified** chi-squared **Test**.

Renamed from ``scPyDE`` (module ``CustomDE.scpyde``) in July 2026: that name
read as "Python SCDE", and SCDE (Kharchenko et al., Nat Methods 2014) is an
established but statistically unrelated method -- a Bayesian dropout error
model, not a hurdle. ``run_scpyde`` and ``run_python_hurdle_de`` remain as
aliases for ``run_duet``.

Why this exists
---------------
R MAST is accurate but, in this pipeline, requires (a) an R + WSL toolchain and
(b) writing a per-cell-type expression matrix to disk before R re-reads it.
This module removes both costs: it operates directly on the in-memory AnnData,
never densifies the full matrix, streams one sparse gene-column at a time, and
writes results chunk-by-chunk so peak RAM stays flat regardless of gene count.

It does **not** reproduce R MAST bit-for-bit. It follows the same high-level
hurdle logic (a logistic *detection* model + a Gaussian *positive-expression*
model, combined by summing their likelihood-ratio statistics) and emits the
same output columns the existing pipeline already consumes.

Statistical model (per gene, per cell type)
--------------------------------------------
Let ``y`` be the log-normalized expression of one gene across the cells of one
cell type, and ``cond`` the condition indicator (0 = reference, 1 = test).

Part 1 — Detection (logistic):
    y_detect = 1[y > 0]
    full:  y_detect ~ 1 + cond + CDR (+ covariates)
    null:  y_detect ~ 1        + CDR (+ covariates)
    stat_detect = 2 * (loglik_full - loglik_null)        (>= 0)
    df_detect   = number of condition columns            (1 for 2 groups)
    p_detect    = chi2.sf(stat_detect, df_detect)

Part 2 — Positive expression (Gaussian OLS), on cells with y > 0 only:
    full:  y ~ 1 + cond + CDR (+ covariates)
    null:  y ~ 1        + CDR (+ covariates)
    stat_continuous = n_pos * ln(RSS_null / RSS_full)    (Gaussian LRT, >= 0)
    df_continuous   = number of condition columns
    p_continuous    = chi2.sf(stat_continuous, df_continuous)
    coef_continuous = condition coefficient from the *full* continuous model

The reported ``coef`` column defaults to the MAST-style hurdle logFC -- the
marginal mean difference ``mean_expr_test - mean_expr_ref`` (= detection-rate x
positive-mean difference) -- which matches R MAST's ``coef`` essentially
exactly and the downstream ``|logFC|>0.25`` thresholds. Pass ``linear_logfc=True`` (or ``coef_mode="mast_logfc_linear"``, CLI
``--linear-logfc``) for the scanpy/Wilcoxon-style LINEAR log2 fold change (the same
marginal effect back-transformed to the linear scale, so its magnitude matches
diffxpy/DESeq2/Wilcoxon instead of being log1p-compressed). The switch defaults to
COMPRESSED so ``coef`` resembles R MAST out of the box. ``coef_mode="continuous"``
instead reports ``coef_continuous``.

Combined hurdle test:
    stat_hurdle = stat_detect + stat_continuous
    df_hurdle   = df_detect   + df_continuous
    pvalue      = chi2.sf(stat_hurdle, df_hurdle)
    neglog10p   = -log10(pvalue), computed WITHOUT going through the float64
                  p-value, so it stays exact past the underflow floor
    fdr         = Benjamini-Hochberg over *tested* genes within each
                  (comparison, cell type) block.

Ranking note: at tens of thousands of cells the hurdle statistic routinely exceeds
~1400, where ``chi2.sf`` underflows to exactly 0.0 -- roughly 3-7% of genes on an
11k-cell type. Sorting by ``pvalue`` therefore ties all of them (their true
-log10 p spans 311 to 3258), which shifts top-K gene lists by tens of genes.
**Rank on ``neglog10p``, not on ``pvalue``.** ``pvalue`` and ``Pr(>Chisq)`` keep
their underflowed zeros so the MAST-compatible output contract is unchanged.

CDR (cellular detection rate) = (# nonzero genes in a cell) / (# genes), used
as the default nuisance covariate. It is z-scored per cell type purely for
numerical conditioning (this does not change the condition LRT or coefficient).

See ``README_python_hurdle.md`` for the full assumptions/differences write-up.
"""

from __future__ import annotations

import datetime
import gc
import math
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2, chi2_contingency, fisher_exact, t as _student_t
from scipy.special import digamma, gammaln, polygamma, xlogy

__version__ = "1.0.0"

# --------------------------------------------------------------------------- #
# Optional dependencies (kept lazy / soft so importing this module is cheap).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - trivial
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **_kwargs):  # type: ignore
        return iterable if iterable is not None else []

try:  # pragma: no cover
    import psutil

    _PROC = psutil.Process(os.getpid())
except Exception:  # pragma: no cover
    psutil = None
    _PROC = None


# --------------------------------------------------------------------------- #
# Output schema. Order is fixed and deterministic.
# --------------------------------------------------------------------------- #
HURDLE_OUTPUT_COLUMNS = [
    # --- MAST-compatible core (downstream code reads these) ---
    "primerid",          # == gene  (MAST compatibility)
    "gene",
    "coef",              # MAST-style hurdle logFC (marginal mean diff) by default
    "coef_continuous",   # continuous positive-expression condition coefficient
    "Pr(>Chisq)",        # == pvalue (MAST compatibility)
    "pvalue",
    "fdr",
    "neglog10p",         # underflow-safe -log10(pvalue) -- USE THIS FOR RANKING
    # --- hurdle decomposition ---
    "p_detect",
    "p_continuous",
    "stat_detect",
    "stat_continuous",
    "stat_hurdle",
    "df_detect",
    "df_continuous",
    "df_hurdle",
    # --- descriptive counts (computed even for skipped genes) ---
    "n_cells",
    "n_ref",
    "n_test",
    "n_positive",
    "n_positive_ref",
    "n_positive_test",
    "detect_rate_ref",
    "detect_rate_test",
    "mean_expr_ref",
    "mean_expr_test",
    # --- grouping / provenance ---
    "celltype",
    "comparison",
    "ref_level",
    "test_level",
    "lrt_term",           # MAST-compatible: e.g. "conditionpancreas"
    "tested",
    "skip_reason",
    "method",
]

METHOD_NAME = "DUET"

# Skip-reason vocabulary (kept stable for downstream filtering / QC).
SKIP_ALL_ZERO = "all_zero"
SKIP_ONE_GROUP_MISSING = "one_group_missing"
SKIP_TOO_FEW_CELLS = "too_few_cells"
SKIP_TOO_FEW_POSITIVE = "too_few_positive"
SKIP_NO_POS_ONE_GROUP = "no_positive_in_one_group"
SKIP_SINGULAR = "singular_design"
SKIP_SEPARATION = "logistic_separation"
SKIP_MODEL_FAILED = "model_failed"


# --------------------------------------------------------------------------- #
# Logging helpers.
# --------------------------------------------------------------------------- #
_T0 = time.time()


def _ph_log(celltype: str, message: str) -> None:
    """Pipeline-style log line: ``[PY-HURDLE][celltype][time] ...``."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - _T0
    print(f"[PY-HURDLE][{celltype}][{now}][{elapsed:7.1f}s] {message}", flush=True)


def _log_mem(celltype: str, tag: str, enabled: bool) -> None:
    if not enabled or _PROC is None:
        return
    try:
        rss_gb = _PROC.memory_info().rss / (1024 ** 3)
        _ph_log(celltype, f"[MEM] {tag}: RSS={rss_gb:.2f} GB")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Sparse-aware low-level numeric helpers.
# --------------------------------------------------------------------------- #
def _column_dense(matrix, j: int) -> np.ndarray:
    """Return column ``j`` as a 1-D dense array WITHOUT densifying the matrix."""
    if sp.issparse(matrix):
        return np.asarray(matrix.getcol(j).todense()).ravel()
    return np.asarray(matrix[:, j]).ravel()


def _compute_cdr(matrix, n_total_genes: int) -> np.ndarray:
    """Cellular detection rate per cell = (# positive genes) / n_total_genes.

    Counts *truly positive* entries (value > 0). For sparse input this never
    densifies and never mutates the source matrix.
    """
    n_cells = matrix.shape[0]
    if sp.issparse(matrix):
        csc = matrix.tocsc()
        pos = csc.data > 0
        # csc.indices holds the row (cell) index of each stored value.
        counts = np.bincount(csc.indices[pos], minlength=n_cells).astype(np.float64)
    else:
        counts = (np.asarray(matrix) > 0).sum(axis=1).astype(np.float64)
    return counts / float(max(1, n_total_genes))


def _zscore_columns(block: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Z-score numeric covariate columns; drop constant columns (singular guard).

    Returns the kept columns and their (possibly reduced) names. Scaling a
    nuisance covariate does not change the condition LRT or coefficient; it only
    improves conditioning of the design matrix.
    """
    if block.size == 0:
        return block, names
    keep_cols, keep_names = [], []
    for k in range(block.shape[1]):
        col = block[:, k].astype(np.float64)
        std = col.std()
        if std < 1e-12:
            # Constant covariate carries no information and would make the
            # design rank-deficient -> drop it.
            continue
        keep_cols.append((col - col.mean()) / std)
        keep_names.append(names[k])
    if not keep_cols:
        return np.empty((block.shape[0], 0)), []
    return np.column_stack(keep_cols), keep_names


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR. NaN p-values pass through as NaN."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return out
    pf = p[finite]
    order = np.argsort(pf, kind="mergesort")
    ranked = pf[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    adj = np.empty(m)
    adj[order] = q
    out[finite] = adj
    return out


# --------------------------------------------------------------------------- #
# Logistic detection component.
# --------------------------------------------------------------------------- #
def _logit_irls(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-8):
    """Compact IRLS logistic fit (NumPy). Returns dict or None on singular solve.

    Used as the fast engine and as a fallback when statsmodels is unavailable.
    """
    n, p = X.shape
    beta = np.zeros(p)
    converged = False
    for it in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-10, None)
        z = eta + (y - mu) / w
        xw = X * w[:, None]
        a = X.T @ xw
        b = X.T @ (w * z)
        try:
            beta_new = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(beta_new)):
            return None
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    eta = np.clip(X @ beta, -30.0, 30.0)
    mu = 1.0 / (1.0 + np.exp(-eta))
    eps = 1e-12
    llf = float(np.sum(y * np.log(mu + eps) + (1.0 - y) * np.log(1.0 - mu + eps)))
    max_abs = float(np.max(np.abs(beta))) if beta.size else 0.0
    return {
        "llf": llf,
        "beta": beta,
        "converged": converged,
        # Heuristic separation flag: coefficients diverging or non-convergence.
        "separated": (max_abs > 25.0) or (not converged),
    }


def _logit_statsmodels(X: np.ndarray, y: np.ndarray):
    """statsmodels GLM Binomial fit. Returns dict or None on failure/separation."""
    try:
        import statsmodels.api as sm
        from statsmodels.tools.sm_exceptions import PerfectSeparationError
    except Exception:
        return _logit_irls(X, y)  # graceful degradation

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=100)
        llf = float(res.llf)
        beta = np.asarray(res.params, dtype=np.float64)
        if not np.isfinite(llf) or not np.all(np.isfinite(beta)):
            return None
        max_abs = float(np.max(np.abs(beta))) if beta.size else 0.0
        return {
            "llf": llf,
            "beta": beta,
            "converged": True,
            "separated": max_abs > 25.0,
        }
    except PerfectSeparationError:
        return None
    except Exception:
        return None


def _detection_lrt(X_full, X_null, y_detect, n_cond_cols, engine, fallback, cond01):
    """Detection-component LRT.

    Returns (stat, df, pvalue, status) where status is one of:
    'ok', 'constant' (no detection variation -> 0 df), 'fallback', 'failed'.
    """
    s = float(y_detect.sum())
    n = y_detect.size
    # No detection variation: detection carries zero information.
    # The hurdle then reduces to the continuous component (df_detect = 0).
    if s == 0 or s == n:
        return 0.0, 0, np.nan, "constant"

    fit = _logit_statsmodels if engine == "statsmodels" else _logit_irls
    rf = fit(X_full, y_detect)
    rn = fit(X_null, y_detect)

    good = (
        rf is not None
        and rn is not None
        and not rf.get("separated", False)
        and not rn.get("separated", False)
        and np.isfinite(rf["llf"])
        and np.isfinite(rn["llf"])
    )
    if good:
        stat = max(0.0, 2.0 * (rf["llf"] - rn["llf"]))
        df = int(n_cond_cols)
        return stat, df, float(chi2.sf(stat, df)) if df > 0 else np.nan, "ok"

    # --- fallback path (ignores covariates; documented approximation) ---
    if fallback in ("chisq", "fisher"):
        det = (y_detect > 0).astype(int)
        # 2x2 table: rows = condition (ref/test), cols = detected (0/1)
        a = int(np.sum((cond01 == 0) & (det == 0)))
        b = int(np.sum((cond01 == 0) & (det == 1)))
        c = int(np.sum((cond01 == 1) & (det == 0)))
        d = int(np.sum((cond01 == 1) & (det == 1)))
        table = np.array([[a, b], [c, d]])
        # Degenerate margins -> no testable detection effect.
        if table.sum(0).min() == 0 or table.sum(1).min() == 0:
            return 0.0, 0, np.nan, "constant"
        try:
            if fallback == "fisher":
                _, pval = fisher_exact(table)
                stat = float(chi2.isf(min(max(pval, 1e-300), 1.0), 1))
            else:
                stat, pval, _, _ = chi2_contingency(table, correction=False)
                stat = float(stat)
            return max(0.0, stat), 1, float(pval), "fallback"
        except Exception:
            return np.nan, 0, np.nan, "failed"

    return np.nan, 0, np.nan, "failed"


# --------------------------------------------------------------------------- #
# Empirical-Bayes variance moderation (limma squeezeVar; Smyth 2004).
# --------------------------------------------------------------------------- #
def _trigamma_inverse(x: float) -> float:
    """Solve trigamma(y) = x for y (Newton iteration; limma's trigammaInverse)."""
    if not np.isfinite(x):
        return np.nan
    if x > 1e7:
        return 1.0 / math.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = polygamma(1, y)
        dif = tri * (1.0 - tri / x) / polygamma(2, y)
        y += dif
        if -dif / y < 1e-8:
            break
    return float(y)


def _squeeze_var(s2: np.ndarray, df: np.ndarray):
    """Estimate the inverse-chi-square prior (d0, s0^2) from per-gene variances.

    Mirrors limma::fitFDist / squeezeVar (Smyth, 2004). Returns (d0, s0_sq);
    d0 may be +inf (degenerate -> full shrinkage to s0_sq).
    """
    s2 = np.asarray(s2, dtype=np.float64)
    df = np.asarray(df, dtype=np.float64)
    ok = np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 0)
    s2, df = s2[ok], df[ok]
    if s2.size < 2:
        return 0.0, (float(np.mean(s2)) if s2.size else 1.0)
    z = np.log(s2)
    e = z - digamma(df / 2.0) + np.log(df / 2.0)
    emean = float(np.mean(e))
    n = e.size
    evar = float(np.sum((e - emean) ** 2) / (n - 1) - np.mean(polygamma(1, df / 2.0)))
    if evar > 0:
        d0_half = _trigamma_inverse(evar)
        d0 = 2.0 * d0_half
        s0_sq = float(np.exp(emean + digamma(d0_half) - np.log(d0_half)))
    else:
        d0 = np.inf  # degenerate: shrink fully to the prior scale
        s0_sq = float(np.exp(emean))
    return d0, s0_sq


# --------------------------------------------------------------------------- #
# Gaussian positive-expression component.
# --------------------------------------------------------------------------- #
def _continuous_lrt(X_full_pos, X_null_pos, y_pos, cond_col_index, n_cond_cols):
    """Gaussian LRT on positive cells.

    Returns a dict with the LRT result plus the pieces needed for empirical-Bayes
    moderation: ``s2`` (residual variance), ``dfresid`` (residual dof), ``vcoef``
    (unscaled variance of the condition coefficient, [(X'X)^-1]_{cond,cond}).
    On failure returns ``status`` in {singular_design, model_failed}.
    """
    fail = {"stat": np.nan, "df": 0, "p": np.nan, "coef": np.nan,
            "s2": np.nan, "dfresid": np.nan, "vcoef": np.nan}
    n_pos, p_full = X_full_pos.shape
    if n_pos <= p_full:
        return {**fail, "status": SKIP_SINGULAR}

    rank = np.linalg.matrix_rank(X_full_pos)
    if rank < p_full:
        return {**fail, "status": SKIP_SINGULAR}

    try:
        beta_f, _, _, _ = np.linalg.lstsq(X_full_pos, y_pos, rcond=None)
        beta_n, _, _, _ = np.linalg.lstsq(X_null_pos, y_pos, rcond=None)
        xtx_inv = np.linalg.inv(X_full_pos.T @ X_full_pos)
    except np.linalg.LinAlgError:
        return {**fail, "status": SKIP_MODEL_FAILED}

    resid_f = y_pos - X_full_pos @ beta_f
    resid_n = y_pos - X_null_pos @ beta_n
    rss_f = float(resid_f @ resid_f)
    rss_n = float(resid_n @ resid_n)
    coef = float(beta_f[cond_col_index])
    if not np.isfinite(rss_f) or not np.isfinite(rss_n):
        return {**fail, "coef": coef, "status": SKIP_MODEL_FAILED}

    dfresid = int(n_pos - p_full)
    s2 = (rss_f / dfresid) if dfresid > 0 else np.nan
    vcoef = float(xtx_inv[cond_col_index, cond_col_index])

    rss_f = max(rss_f, 1e-12)
    stat = 0.0 if rss_n <= rss_f else float(n_pos * math.log(rss_n / rss_f))
    stat = max(0.0, stat)
    df = int(n_cond_cols)
    pval = float(chi2.sf(stat, df)) if df > 0 else np.nan
    return {"stat": stat, "df": df, "p": pval, "coef": coef,
            "s2": s2, "dfresid": dfresid, "vcoef": vcoef, "status": "ok"}


def _apply_eb_shrinkage(df: pd.DataFrame, celltype: str, eb_min_genes: int = 10) -> pd.DataFrame:
    """Cell-type-level empirical-Bayes moderation of the continuous component.

    Replaces each gene's continuous p-value with a moderated t-test that uses the
    shrunken residual variance (posterior of the inverse-chi-square prior fit
    across genes), then recombines the hurdle statistic. Detection-only genes
    (no continuous fit) are left untouched. This is what brings the per-gene
    variances -- and hence the FDR calibration -- close to R MAST/limma.
    """
    if "_eb_dfresid" not in df.columns:
        return df
    mask = (
        df["_eb_dfresid"].notna() & (df["_eb_dfresid"] > 0)
        & df["_eb_s2"].notna() & (df["_eb_s2"] > 0)
        & df["_eb_vcoef"].notna() & (df["_eb_vcoef"] > 0)
        & df["coef_continuous"].notna()
    ).to_numpy()
    if int(mask.sum()) < eb_min_genes:
        _ph_log(celltype, f"EB shrinkage skipped: only {int(mask.sum())} genes with a continuous fit.")
        return df

    s2 = df.loc[mask, "_eb_s2"].to_numpy(float)
    dfr = df.loc[mask, "_eb_dfresid"].to_numpy(float)
    vcoef = df.loc[mask, "_eb_vcoef"].to_numpy(float)
    beta = df.loc[mask, "coef_continuous"].to_numpy(float)
    stat_d = np.nan_to_num(df.loc[mask, "stat_detect"].to_numpy(float), nan=0.0)
    df_d = np.nan_to_num(df.loc[mask, "df_detect"].to_numpy(float), nan=0.0)

    d0, s0_sq = _squeeze_var(s2, dfr)
    d0c = min(d0, 1e6)  # cap +inf for a finite posterior dof
    s2_post = (d0c * s0_sq + dfr * s2) / (d0c + dfr)
    post_df = dfr + d0c
    se = np.sqrt(np.maximum(s2_post * vcoef, 1e-300))
    tval = beta / se
    p_c = np.clip(2.0 * _student_t.sf(np.abs(tval), post_df), 1e-300, 1.0)
    stat_c = chi2.isf(p_c, 1)                     # 1-df chi-square, additive in hurdle
    stat_h = stat_d + stat_c
    df_h = (df_d + 1).astype(int)
    pval = chi2.sf(stat_h, df_h)

    idx = df.index[mask]
    df.loc[idx, "p_continuous"] = p_c
    df.loc[idx, "stat_continuous"] = stat_c
    df.loc[idx, "df_continuous"] = 1
    df.loc[idx, "stat_hurdle"] = stat_h
    df.loc[idx, "df_hurdle"] = df_h
    df.loc[idx, "pvalue"] = pval
    df.loc[idx, "Pr(>Chisq)"] = pval
    _ph_log(celltype, f"EB shrinkage: prior df0={d0:.2f} s0^2={s0_sq:.4g} moderated {int(mask.sum())} genes")
    return df


# --------------------------------------------------------------------------- #
# Per-gene fit.
# --------------------------------------------------------------------------- #
def _blank_row(gene, ct, comparison, ref_label, test_label):
    row = {c: np.nan for c in HURDLE_OUTPUT_COLUMNS}
    row.update(
        primerid=gene,
        gene=gene,
        celltype=ct,
        comparison=comparison,
        ref_level=ref_label,
        test_level=test_label,
        lrt_term=f"condition{test_label}",
        tested=False,
        skip_reason="",
        method=METHOD_NAME,
    )
    return row


def _fit_one_gene(col, ctx) -> dict:
    """Fit the hurdle model for a single gene-column ``col`` (1-D dense array)."""
    gene = ctx["gene_name"]
    row = _blank_row(gene, ctx["celltype"], ctx["comparison"], ctx["ref_label"], ctx["test_label"])

    cond01 = ctx["cond01"]
    ref_mask = ctx["ref_mask"]
    test_mask = ctx["test_mask"]
    n_ref = ctx["n_ref"]
    n_test = ctx["n_test"]

    col = col.astype(np.float64, copy=False)
    if ctx["apply_log1p"]:
        col = np.log1p(col)

    pos_mask = col > 0
    n_pos = int(pos_mask.sum())
    pos_ref = int(np.sum(pos_mask & ref_mask))
    pos_test = int(np.sum(pos_mask & test_mask))

    # Descriptive stats are filled even when the gene is ultimately skipped.
    row.update(
        n_cells=int(col.size),
        n_ref=n_ref,
        n_test=n_test,
        n_positive=n_pos,
        n_positive_ref=pos_ref,
        n_positive_test=pos_test,
        detect_rate_ref=(pos_ref / n_ref) if n_ref else np.nan,
        detect_rate_test=(pos_test / n_test) if n_test else np.nan,
        mean_expr_ref=float(col[ref_mask].mean()) if n_ref else np.nan,
        mean_expr_test=float(col[test_mask].mean()) if n_test else np.nan,
    )

    # MAST-style logFC: marginal hurdle effect = E_test[Y] - E_ref[Y].
    # With a ~condition model this equals (mean over all test cells) minus
    # (mean over all ref cells), i.e. detection-rate x positive-mean difference.
    marginal_logfc = row["mean_expr_test"] - row["mean_expr_ref"]
    partial = ctx["partial_hurdle"]

    # all-zero is always a hard skip
    if n_pos == 0:
        row["skip_reason"] = SKIP_ALL_ZERO
        return row

    # Strict mode keeps the original (conservative) gates and ordering so that
    # behaviour/skip_reasons are unchanged when partial_hurdle=False.
    if not partial:
        if n_pos < ctx["min_positive"]:
            row["skip_reason"] = SKIP_TOO_FEW_POSITIVE
            return row
        if pos_ref == 0 or pos_test == 0:
            row["skip_reason"] = SKIP_NO_POS_ONE_GROUP
            return row

    # ---- Part 1: detection (all cells) ----
    y_detect = pos_mask.astype(np.float64)
    stat_d, df_d, p_d, status_d = _detection_lrt(
        ctx["X_full_det"], ctx["X_null_det"], y_detect,
        ctx["n_cond_cols"], ctx["logistic_engine"], ctx["detection_fallback"], cond01,
    )
    if status_d == "failed":
        if not partial:
            row["skip_reason"] = SKIP_SEPARATION
            return row
        stat_d, df_d, p_d = 0.0, 0, np.nan  # drop detection; keep continuous

    # ---- Part 2: continuous (positive cells only) ----
    coef_cont = np.nan
    stat_c, df_c, p_c = 0.0, 0, np.nan
    can_attempt_cont = (pos_ref >= 1 and pos_test >= 1 and n_pos >= ctx["min_positive"])
    if can_attempt_cont:
        pos_idx = np.flatnonzero(pos_mask)
        Xf_pos = ctx["X_full_cont"][pos_idx]
        Xn_pos = ctx["X_null_cont"][pos_idx]
        y_pos = col[pos_idx]
        cres = _continuous_lrt(Xf_pos, Xn_pos, y_pos, ctx["cond_col_index"], ctx["n_cond_cols"])
        if cres["status"] in (SKIP_SINGULAR, SKIP_MODEL_FAILED):
            if not partial:
                row["skip_reason"] = cres["status"]
                return row
            # partial mode: continuous contributes nothing
        else:
            stat_c, df_c, p_c, coef_cont = cres["stat"], cres["df"], cres["p"], cres["coef"]
            if ctx["eb_shrinkage"]:
                # stash pieces for the cell-type-level variance moderation pass
                row["_eb_s2"] = cres["s2"]
                row["_eb_dfresid"] = cres["dfresid"]
                row["_eb_vcoef"] = cres["vcoef"]
    # else (partial mode only): continuous contributes nothing (detection-only gene)

    # ---- combine ----
    df_h = int(df_d + df_c)
    if df_h <= 0:
        # Nothing estimable (e.g. detected in every cell AND positives in one group only).
        row["skip_reason"] = SKIP_MODEL_FAILED
        return row
    stat_h = float(stat_d + stat_c)
    pvalue = float(chi2.sf(stat_h, df_h))
    neglog10p = float(_neglog10_chi2_sf(np.array([stat_h]), np.array([float(df_h)]))[0])

    if ctx["coef_mode"] == "mast_logfc_linear":
        coef_out = _linear_logfc(row["mean_expr_test"], row["mean_expr_ref"])
    elif ctx["coef_mode"] == "continuous":
        coef_out = coef_cont
    else:
        coef_out = marginal_logfc
    row.update(
        coef=float(coef_out) if np.isfinite(coef_out) else np.nan,
        coef_continuous=float(coef_cont) if np.isfinite(coef_cont) else np.nan,
        pvalue=pvalue,
        neglog10p=neglog10p,
        p_detect=p_d,
        p_continuous=p_c,
        stat_detect=float(stat_d),
        stat_continuous=float(stat_c),
        stat_hurdle=stat_h,
        df_detect=int(df_d),
        df_continuous=int(df_c),
        df_hurdle=df_h,
        tested=True,
        skip_reason="",
    )
    row["Pr(>Chisq)"] = pvalue  # MAST compatibility (== pvalue)
    return row


# --------------------------------------------------------------------------- #
# Worker plumbing for chunked / parallel fitting.
# --------------------------------------------------------------------------- #
# A module-level slot lets fork (Linux) share the per-cell-type context via
# copy-on-write, and lets spawn (Windows) pickle it ONCE per worker (via the
# pool initializer) rather than once per chunk.
_WORKER_CTX: dict | None = None


def _worker_init(shared: dict) -> None:  # pragma: no cover - exercised in subprocess
    global _WORKER_CTX
    _WORKER_CTX = shared


def _fit_chunk_indices(start: int, end: int) -> list[dict]:
    """Fit genes [start, end) using the worker's stashed context."""
    return _fit_chunk_with_ctx(_WORKER_CTX, start, end)


def _fit_chunk_with_ctx(shared: dict, start: int, end: int) -> list[dict]:
    matrix = shared["matrix"]
    var_names = shared["var_names"]
    base = shared["gene_ctx"]
    rows = []
    for g in range(start, end):
        col = _column_dense(matrix, g)
        gene_ctx = dict(base)
        gene_ctx["gene_name"] = str(var_names[g])
        rows.append(_fit_one_gene(col, gene_ctx))
    return rows


# --------------------------------------------------------------------------- #
# Streaming result writer (CSV append, or parquet via pyarrow row-groups).
# --------------------------------------------------------------------------- #
class _ResultWriter:
    """Write result blocks incrementally so the full table never lives in RAM."""

    def __init__(self, path: str, fmt: str = "csv"):
        self.path = path
        self.fmt = fmt
        self._csv_header_written = False
        self._pq_writer = None
        self._wrote_any = False
        if fmt == "csv" and os.path.exists(path):
            os.remove(path)

    def write_block(self, df: pd.DataFrame) -> None:
        df = df.reindex(columns=HURDLE_OUTPUT_COLUMNS)
        self._wrote_any = True
        if self.fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._pq_writer is None:
                self._pq_writer = pq.ParquetWriter(self.path, table.schema)
            self._pq_writer.write_table(table)
        else:
            df.to_csv(
                self.path,
                mode="w" if not self._csv_header_written else "a",
                header=not self._csv_header_written,
                index=False,
            )
            self._csv_header_written = True

    def close(self) -> None:
        # Always leave a readable, schema-correct file behind, even if no
        # cell type produced rows (downstream code does os.path.exists checks
        # and then read_csv -> a header-only file must exist).
        if not self._wrote_any:
            empty = pd.DataFrame(columns=HURDLE_OUTPUT_COLUMNS)
            if self.fmt == "parquet":
                empty.to_parquet(self.path, index=False)
            else:
                empty.to_csv(self.path, index=False)
        if self._pq_writer is not None:
            self._pq_writer.close()


# --------------------------------------------------------------------------- #
# Design-matrix construction (built once per cell type, reused across genes).
# --------------------------------------------------------------------------- #
def _build_designs(cond01, cdr, obs_ct, covariates):
    """Construct detection/continuous full & null designs for one cell type.

    Column layout (full):  [intercept, condition, <covariates...>]
    Column layout (null):  [intercept,            <covariates...>]
    The detection and continuous designs are structurally identical; the
    continuous fit simply indexes the positive-cell rows per gene.

    Returns (X_full, X_null, cond_col_index, n_cond_cols, used_covariate_names).
    """
    n = cond01.size
    intercept = np.ones((n, 1), dtype=np.float64)
    cond_col = cond01.astype(np.float64).reshape(-1, 1)

    # Assemble covariate block.
    cov_cols, cov_names = [], []
    for name in covariates:
        if str(name).upper() == "CDR":
            cov_cols.append(cdr.astype(np.float64))
            cov_names.append("CDR")
        elif obs_ct is not None and name in obs_ct.columns:
            series = obs_ct[name]
            if pd.api.types.is_numeric_dtype(series):
                cov_cols.append(pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64))
                cov_names.append(name)
            else:
                # Categorical -> one-hot, drop first level (reference).
                dummies = pd.get_dummies(series.astype(str), prefix=name, drop_first=True)
                for c in dummies.columns:
                    cov_cols.append(dummies[c].to_numpy(dtype=np.float64))
                    cov_names.append(c)
        # silently ignore unknown covariate names (logged by caller)

    if cov_cols:
        cov_block = np.column_stack(cov_cols)
        # Impute any NaN covariate values with the column mean before scaling.
        col_means = np.nanmean(cov_block, axis=0)
        inds = np.where(np.isnan(cov_block))
        if inds[0].size:
            cov_block[inds] = np.take(col_means, inds[1])
        cov_block, cov_names = _zscore_columns(cov_block, cov_names)
    else:
        cov_block = np.empty((n, 0))

    X_full = np.hstack([intercept, cond_col, cov_block])
    X_null = np.hstack([intercept, cov_block])
    cond_col_index = 1            # condition sits right after the intercept
    n_cond_cols = 1               # 2-group comparison -> single df
    return X_full, X_null, cond_col_index, n_cond_cols, cov_names


def _linear_logfc(mean_test, mean_ref, eps=1e-9):
    """scanpy/Wilcoxon-style linear-scale log2 fold change for log1p data.

    The MAST-style ``coef`` is the raw difference of per-group log1p means
    (``mean_test - mean_ref``); by Jensen's inequality that is systematically
    COMPRESSED relative to the log-ratio of the group means, which is why DUET's
    log2FC looked weaker than diffxpy/DESeq2/Wilcoxon. Here we back-transform each
    group's log1p mean to the linear (expm1) scale and take log2 of their ratio --
    EXACTLY scanpy ``rank_genes_groups``'s ``logfoldchanges`` definition (same
    ``eps=1e-9``) -- so for co-expressed genes the magnitude matches Wilcoxon to
    numerical precision (Pearson ~1.0) and is on the same scale as diffxpy/DESeq2.

    Note: a gene detected in only ONE group (``expm1(mean)~0`` on the other side)
    gets a large |log2FC| (~tens), exactly as scanpy/Seurat report it -- such genes
    really are ~infinitely up/down. DUET's hurdle flags more of these (it also
    tests the detection component), so its *global* median |log2FC| sits higher than
    Wilcoxon's; on the shared, co-expressed genes (e.g. anything diffxpy tested, and
    the genes shown in the comparison heatmaps) the two agree. Works on scalars/arrays.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2((np.expm1(mean_test) + eps) / (np.expm1(mean_ref) + eps))


def _neglog10_chi2_sf(stat, df):
    """-log10 of the chi2 upper tail, valid past the float64 underflow floor.

    ``chi2.sf`` returns exactly 0.0 -- and ``chi2.logsf`` returns -inf -- once the
    tail drops below ~5e-324, which the hurdle statistic routinely does at tens of
    thousands of cells (stat > ~1400). Ranking genes by ``pvalue`` then ties every
    one of them: on 11k cells this collapses ~7% of genes, whose true -log10 p spans
    311 to 3258, into a single rank. We keep scipy's value wherever it is finite and
    otherwise fall back to the asymptotic expansion of the upper incomplete gamma:

        log sf(x; k) -> -x/2 + (k/2 - 1) * log(x/2) - lgamma(k/2)

    which is EXACT for k=2 (the standard hurdle: sf = exp(-x/2)) and has relative
    error < 1e-6 for k=1 in the regime where the fallback is used. Verified against
    ``chi2.logsf`` over x in [1, 1400] for df 1/2/3: max abs error 0.0.

    ``pvalue`` / ``Pr(>Chisq)`` are deliberately left alone for backward
    compatibility; this is an additional column, not a replacement.
    """
    stat = np.asarray(stat, dtype=float)
    df = np.broadcast_to(np.asarray(df, dtype=float), stat.shape)
    out = np.full(stat.shape, np.nan)
    ok = np.isfinite(stat) & (stat >= 0.0) & (df > 0)
    if not ok.any():
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        ls = chi2.logsf(stat[ok], df[ok])
    x, k = stat[ok], df[ok]
    bad = ~np.isfinite(ls)
    if bad.any():
        ls = np.where(bad, -x / 2.0 + (k / 2.0 - 1.0) * np.log(x / 2.0) - gammaln(k / 2.0), ls)
    out[ok] = -ls / math.log(10.0)
    return out


# --------------------------------------------------------------------------- #
# Per-cell-type driver.
# --------------------------------------------------------------------------- #
def _fit_celltype_vectorized(
    X, var_names, cond01, ref_mask, test_mask, *,
    comparison, ref_label, test_label, min_positive, partial_hurdle,
    eb_shrinkage, apply_log1p, celltype, coef_mode="mast_logfc",
):
    """Fully vectorized two-group hurdle for the no-covariate design.

    With design = [intercept, condition] both components have closed forms:
      * detection  = the 2x2 likelihood-ratio (G-test), identical to the
        logistic LRT for two groups (handles separation exactly, no IRLS);
      * continuous = a two-group Gaussian LRT from per-group sums/sum-of-squares.
    All gene quantities are obtained from THREE sparse reductions over the
    cell-type matrix (sum, count>0, sum-of-squares per group), so there is no
    per-gene Python loop, no iterative solve, and no extra matrix copies sent to
    worker processes. Results match the per-gene path to numerical precision.
    """
    n_cells, n_genes = X.shape
    n_ref = int(np.asarray(ref_mask, bool).sum())
    n_test = int(np.asarray(test_mask, bool).sum())
    n_total = n_ref + n_test

    X = X.tocsc() if sp.issparse(X) else sp.csc_matrix(X)
    data = X.data.astype(np.float64, copy=False)
    if apply_log1p:
        data = np.log1p(data)

    # group indicator (n_cells x 2): col 0 = ref, col 1 = test
    G = np.zeros((n_cells, 2), dtype=np.float64)
    G[np.asarray(ref_mask, bool), 0] = 1.0
    G[np.asarray(test_mask, bool), 1] = 1.0

    # Three reductions; each shares X's indices/indptr, only the data differs.
    sums = sp.csc_matrix((data, X.indices, X.indptr), shape=X.shape).T @ G
    cnts = sp.csc_matrix(((data > 0).astype(np.float64), X.indices, X.indptr), shape=X.shape).T @ G
    sqs = sp.csc_matrix((data * data, X.indices, X.indptr), shape=X.shape).T @ G

    sr, st = sums[:, 0], sums[:, 1]
    cpr, cpt = cnts[:, 0], cnts[:, 1]
    ssr, sst = sqs[:, 0], sqs[:, 1]
    n_pos = cpr + cpt

    safe_cpr = np.where(cpr > 0, cpr, 1.0)
    safe_cpt = np.where(cpt > 0, cpt, 1.0)
    safe_np = np.where(n_pos > 0, n_pos, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_expr_ref = sr / n_ref
        mean_expr_test = st / n_test
        # continuous (positive cells only)
        ss_ref = ssr - np.where(cpr > 0, sr * sr / safe_cpr, 0.0)
        ss_test = sst - np.where(cpt > 0, st * st / safe_cpt, 0.0)
        rss_full = np.maximum(ss_ref + ss_test, 0.0)
        sum_all, ssq_all = sr + st, ssr + sst
        rss_null = np.maximum(ssq_all - np.where(n_pos > 0, sum_all * sum_all / safe_np, 0.0), 0.0)
        beta_cont = np.where(cpt > 0, st / safe_cpt, 0.0) - np.where(cpr > 0, sr / safe_cpr, 0.0)
        dfresid = n_pos - 2.0
        cont_ok = (cpr >= 1) & (cpt >= 1) & (n_pos >= min_positive) & (dfresid >= 1) & (rss_full > 0)
        rss_full_safe = np.maximum(rss_full, 1e-12)
        ratio = np.where(rss_null > rss_full_safe, rss_null / rss_full_safe, 1.0)
        stat_cont = np.maximum(np.where(cont_ok, n_pos * np.log(ratio), 0.0), 0.0)
        s2 = np.where(dfresid >= 1, rss_full / np.where(dfresid >= 1, dfresid, 1.0), np.nan)
        vcoef = np.where((cpr > 0) & (cpt > 0), 1.0 / safe_cpr + 1.0 / safe_cpt, np.nan)

        # detection (2-group G-test == logistic LRT)
        p_ref = cpr / n_ref
        p_test = cpt / n_test
        p0 = n_pos / n_total
        llf_full = (xlogy(cpr, p_ref) + xlogy(n_ref - cpr, 1.0 - p_ref)
                    + xlogy(cpt, p_test) + xlogy(n_test - cpt, 1.0 - p_test))
        llf_null = xlogy(n_pos, p0) + xlogy(n_total - n_pos, 1.0 - p0)
        stat_detect = np.maximum(2.0 * (llf_full - llf_null), 0.0)

    det_varies = (n_pos > 0) & (n_pos < n_total)
    df_detect = np.where(det_varies, 1, 0).astype(np.int64)
    stat_detect = np.where(det_varies, stat_detect, 0.0)
    df_c = np.where(cont_ok, 1, 0).astype(np.int64)
    df_h = df_detect + df_c
    all_zero = (n_pos == 0)

    # p-values for the raw components
    p_cont = np.where(cont_ok, chi2.sf(stat_cont, 1), np.nan)

    # empirical-Bayes variance moderation of the continuous component
    if eb_shrinkage:
        m = cont_ok & np.isfinite(s2) & (s2 > 0) & np.isfinite(vcoef) & (vcoef > 0)
        if int(m.sum()) >= 10:
            d0, s0_sq = _squeeze_var(s2[m], dfresid[m])
            d0c = min(d0, 1e6)
            s2_post = (d0c * s0_sq + dfresid[m] * s2[m]) / (d0c + dfresid[m])
            post_df = dfresid[m] + d0c
            se = np.sqrt(np.maximum(s2_post * vcoef[m], 1e-300))
            tval = beta_cont[m] / se
            pcm = np.clip(2.0 * _student_t.sf(np.abs(tval), post_df), 1e-300, 1.0)
            stat_cont = stat_cont.copy()
            stat_cont[m] = chi2.isf(pcm, 1)
            p_cont[m] = pcm
            _ph_log(celltype, f"EB shrinkage: prior df0={d0:.2f} s0^2={s0_sq:.4g} moderated {int(m.sum())} genes")

    stat_h = np.where(df_detect > 0, stat_detect, 0.0) + np.where(cont_ok, stat_cont, 0.0)

    # skip / tested logic
    skip = np.array([""] * n_genes, dtype=object)
    if partial_hurdle:
        skip[all_zero] = SKIP_ALL_ZERO
        skip[(~all_zero) & (df_h == 0)] = SKIP_MODEL_FAILED
        tested = (~all_zero) & (df_h > 0)
    else:
        skip[all_zero] = SKIP_ALL_ZERO
        skip[(skip == "") & (n_pos < min_positive)] = SKIP_TOO_FEW_POSITIVE
        skip[(skip == "") & ((cpr == 0) | (cpt == 0))] = SKIP_NO_POS_ONE_GROUP
        skip[(skip == "") & (~cont_ok)] = SKIP_SINGULAR
        tested = (skip == "")

    pvalue = np.where(tested, chi2.sf(stat_h, np.where(df_h > 0, df_h, 1)), np.nan)
    p_detect = np.where(tested & (df_detect > 0), chi2.sf(stat_detect, 1), np.nan)
    p_continuous = np.where(tested & cont_ok, p_cont, np.nan)
    tn = lambda a: np.where(tested, a, np.nan)  # noqa: E731  (tested-only mask)

    # `coef` column respects coef_mode: mast_logfc = raw log1p-mean difference (MAST
    # style); mast_logfc_linear = scanpy-style log2(expm1 mean ratio) (comparable
    # magnitude to the other methods); continuous = the positive-cells coefficient.
    if coef_mode == "mast_logfc_linear":
        coef_vals = _linear_logfc(mean_expr_test, mean_expr_ref)
    elif coef_mode == "continuous":
        coef_vals = beta_cont
    else:
        coef_vals = mean_expr_test - mean_expr_ref

    df = pd.DataFrame({
        "primerid": list(map(str, var_names)),
        "gene": list(map(str, var_names)),
        "coef": np.where(tested, coef_vals, np.nan),
        "coef_continuous": np.where(cont_ok, beta_cont, np.nan),
        "Pr(>Chisq)": pvalue,
        "pvalue": pvalue,
        "fdr": np.full(n_genes, np.nan),
        "neglog10p": tn(_neglog10_chi2_sf(stat_h, np.where(df_h > 0, df_h, 1))),
        "p_detect": p_detect,
        "p_continuous": p_continuous,
        "stat_detect": tn(np.where(df_detect > 0, stat_detect, 0.0)),
        "stat_continuous": tn(np.where(cont_ok, stat_cont, 0.0)),
        "stat_hurdle": tn(stat_h),
        "df_detect": tn(df_detect.astype(float)),
        "df_continuous": tn(df_c.astype(float)),
        "df_hurdle": tn(df_h.astype(float)),
        "n_cells": np.full(n_genes, n_cells),
        "n_ref": np.full(n_genes, n_ref),
        "n_test": np.full(n_genes, n_test),
        "n_positive": n_pos.astype(np.int64),
        "n_positive_ref": cpr.astype(np.int64),
        "n_positive_test": cpt.astype(np.int64),
        "detect_rate_ref": cpr / n_ref,
        "detect_rate_test": cpt / n_test,
        "mean_expr_ref": mean_expr_ref,
        "mean_expr_test": mean_expr_test,
        "celltype": celltype,
        "comparison": comparison,
        "ref_level": ref_label,
        "test_level": test_label,
        "lrt_term": f"condition{test_label}",
        "tested": tested,
        "skip_reason": skip,
        "method": METHOD_NAME,
    })
    return df


def _process_celltype(
    ct,
    X_ct,
    var_names,
    cond01,
    obs_ct,
    *,
    n_total_genes,
    comparison,
    ref_label,
    test_label,
    covariates,
    min_positive,
    detection_fallback,
    logistic_engine,
    apply_log1p,
    workers,
    chunk_genes,
    memory_log,
    partial_hurdle,
    coef_mode,
    eb_shrinkage,
    vectorized,
):
    n_cells = X_ct.shape[0]
    n_genes = len(var_names)
    ref_mask = cond01 == 0
    test_mask = cond01 == 1
    n_ref = int(ref_mask.sum())
    n_test = int(test_mask.sum())

    _ph_log(ct, f"cells={n_cells} (ref={n_ref}, test={n_test}) genes={n_genes}")
    _log_mem(ct, "celltype start", memory_log)

    cdr = _compute_cdr(X_ct, n_total_genes)
    X_full, X_null, cond_idx, n_cond_cols, used_covs = _build_designs(
        cond01, cdr, obs_ct, covariates
    )
    _ph_log(ct, f"design built: covariates={['intercept', 'condition'] + used_covs}")

    # ---- FAST PATH: vectorized two-group hurdle when there are no covariates --
    if vectorized and len(used_covs) == 0 and int(n_cond_cols) == 1:
        _ph_log(ct, "fast path: vectorized two-group hurdle (no covariates)")
        df = _fit_celltype_vectorized(
            X_ct, var_names, cond01, ref_mask, test_mask,
            comparison=comparison, ref_label=ref_label, test_label=test_label,
            min_positive=int(min_positive), partial_hurdle=bool(partial_hurdle),
            eb_shrinkage=bool(eb_shrinkage), apply_log1p=bool(apply_log1p), celltype=ct,
            coef_mode=coef_mode,
        )
        tested_mask = df["tested"].astype(bool).to_numpy()
        fdr = np.full(len(df), np.nan)
        if tested_mask.any():
            fdr[tested_mask] = _bh_fdr(df.loc[tested_mask, "pvalue"].to_numpy())
        df["fdr"] = fdr
        df = df.reindex(columns=HURDLE_OUTPUT_COLUMNS)
        del X_full, X_null, cdr
        gc.collect()
        _log_mem(ct, "celltype end", memory_log)
        return df

    # ---- GENERAL PATH: per-gene fitting (needed when covariates are present) --
    # Shared, read-only context reused by every gene in this cell type.
    gene_ctx = {
        "celltype": ct,
        "comparison": comparison,
        "ref_label": ref_label,
        "test_label": test_label,
        "cond01": cond01,
        "ref_mask": ref_mask,
        "test_mask": test_mask,
        "n_ref": n_ref,
        "n_test": n_test,
        "min_positive": int(min_positive),
        "n_cond_cols": int(n_cond_cols),
        "cond_col_index": int(cond_idx),
        "detection_fallback": detection_fallback,
        "logistic_engine": logistic_engine,
        "apply_log1p": bool(apply_log1p),
        "partial_hurdle": bool(partial_hurdle),
        "coef_mode": coef_mode,
        "eb_shrinkage": bool(eb_shrinkage),
        # Detection uses all cells; continuous indexes positive rows per gene.
        "X_full_det": X_full,
        "X_null_det": X_null,
        "X_full_cont": X_full,
        "X_null_cont": X_null,
    }

    chunk_bounds = [(s, min(s + chunk_genes, n_genes)) for s in range(0, n_genes, chunk_genes)]

    rows: list[dict] = []
    if workers and workers > 1 and n_genes > chunk_genes:
        if platform.system() == "Windows":
            _ph_log(
                ct,
                f"WARNING: workers={workers} on Windows uses 'spawn'; the per-"
                f"cell-type matrix is copied into each worker (higher RAM). "
                f"Use workers=1 if memory-constrained.",
            )
        shared = {"matrix": X_ct, "var_names": list(map(str, var_names)), "gene_ctx": gene_ctx}
        _ph_log(ct, f"parallel fit: {len(chunk_bounds)} chunks x ~{chunk_genes} genes, workers={workers}")
        with ProcessPoolExecutor(
            max_workers=int(workers), initializer=_worker_init, initargs=(shared,)
        ) as ex:
            # map() preserves submission order -> deterministic output.
            for block in tqdm(
                ex.map(_fit_chunk_indices, [s for s, _ in chunk_bounds], [e for _, e in chunk_bounds]),
                total=len(chunk_bounds),
                desc=f"[{ct}] gene-chunks",
            ):
                rows.extend(block)
        del shared
    else:
        shared = {"matrix": X_ct, "var_names": list(map(str, var_names)), "gene_ctx": gene_ctx}
        for (s, e) in tqdm(chunk_bounds, desc=f"[{ct}] gene-chunks"):
            rows.extend(_fit_chunk_with_ctx(shared, s, e))
        del shared

    # Build with any helper columns present (EB pieces), then moderate, then
    # FDR, then restrict to the public schema.
    df = pd.DataFrame(rows)

    if eb_shrinkage:
        df = _apply_eb_shrinkage(df, ct)

    # FDR over tested genes within this (comparison, cell type) block.
    tested_mask = df["tested"].astype(bool).to_numpy()
    fdr = np.full(len(df), np.nan)
    if tested_mask.any():
        fdr[tested_mask] = _bh_fdr(df.loc[tested_mask, "pvalue"].to_numpy())
    df["fdr"] = fdr
    df = df.reindex(columns=HURDLE_OUTPUT_COLUMNS)  # drop EB helper columns, fix order

    # Free the heavy objects before returning.
    del X_full, X_null, cdr, gene_ctx, rows
    gc.collect()
    _log_mem(ct, "celltype end", memory_log)
    return df


# --------------------------------------------------------------------------- #
# Condition derivation.
# --------------------------------------------------------------------------- #
def _derive_condition(obs, sample_col, ref_label, test_label, condition_col):
    """Return an int8 array: 0 = reference, 1 = test.

    Mirrors the existing pipeline rule (a cell is reference iff ``ref_label`` is
    a case-insensitive substring of its label; everything else is the test
    group). If ``condition_col`` is provided it is used as the label source.
    """
    src_col = condition_col if (condition_col and condition_col in obs.columns) else sample_col
    if src_col not in obs.columns:
        raise KeyError(f"Neither condition_col nor sample_col ('{src_col}') found in adata.obs")
    labels = obs[src_col].astype(str)
    is_ref = labels.str.contains(str(ref_label), case=False, na=False)
    if test_label:
        is_test = labels.str.contains(str(test_label), case=False, na=False)
        # Resolve cells matching neither / both: neither -> test (pipeline rule),
        # both -> reference (ref takes precedence).
        cond = np.where(is_ref, 0, 1).astype(np.int8)
    else:
        cond = np.where(is_ref, 0, 1).astype(np.int8)
    return cond


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #
def _run_hurdle_core(
    adata,
    *,
    cond_all,
    celltype_col,
    output_dir,
    output_prefix,
    comparison,
    ref_label,
    test_label,
    min_cells_per_group,
    min_positive,
    covariates,
    workers,
    chunk_genes,
    detection_fallback,
    logistic_engine,
    assume_logged,
    emit_skipped_celltypes,
    output_format,
    memory_log,
    min_total_cells,
    partial_hurdle,
    coef_mode,
    eb_shrinkage,
    output_name,
    vectorized,
    drop_untested,
):
    """Shared engine behind both the Reference-vs-test and CNV entry points."""
    os.makedirs(output_dir, exist_ok=True)

    if celltype_col not in adata.obs.columns:
        raise KeyError(f"'{celltype_col}' not found in adata.obs")

    var_names = [str(g) for g in adata.var_names]
    n_total_genes = adata.n_vars
    obs = adata.obs
    X = adata.X

    # Detect whether values look unlogged (mirrors get_logged_expr_df()).
    apply_log1p = False
    if assume_logged is None:
        try:
            gmax = float(X.data.max()) if (sp.issparse(X) and X.nnz) else (
                float(np.asarray(X).max()) if not sp.issparse(X) else 0.0
            )
        except Exception:
            gmax = 0.0
        if gmax > 100:
            apply_log1p = True
            print("[PY-HURDLE] WARNING: expression appears unlogged (max>100); "
                  "applying log1p per gene-column.", flush=True)
    else:
        apply_log1p = not bool(assume_logged)

    if output_name:
        # Exact output filename (used by the pipeline to write mast_all_celltypes.csv).
        ext = "parquet" if output_name.lower().endswith(".parquet") else "csv"
        out_path = os.path.join(output_dir, output_name)
        _base = os.path.splitext(os.path.basename(output_name))[0]
        diag_path = os.path.join(output_dir, f"{_base}_diagnostics.csv")
    else:
        ext = "parquet" if str(output_format).lower() == "parquet" else "csv"
        out_path = os.path.join(output_dir, f"{output_prefix}_all_celltypes.{ext}")
        diag_path = os.path.join(output_dir, f"{output_prefix}_diagnostics.csv")
    writer = _ResultWriter(out_path, fmt=ext)

    celltypes = sorted(pd.Index(obs[celltype_col].astype(str)).unique().tolist())
    _ph_log("setup", f"comparison='{comparison}' celltypes={len(celltypes)} "
                     f"genes={n_total_genes} workers={workers} chunk_genes={chunk_genes} "
                     f"engine={logistic_engine} fallback={detection_fallback} "
                     f"eb_shrinkage={eb_shrinkage} fmt={ext}")
    if workers and workers > 1 and platform.system() == "Windows":
        _ph_log("setup", "Windows + workers>1: expect higher RAM (no fork/COW).")

    diag_rows = []
    ct_labels = obs[celltype_col].astype(str).to_numpy()
    total_written = 0

    for ct in celltypes:
        t_ct = time.time()
        mask_ct = ct_labels == ct
        cond_ct = cond_all[mask_ct]
        n_ref = int(np.sum(cond_ct == 0))
        n_test = int(np.sum(cond_ct == 1))

        # ---- cell-type level gates ----
        # R MAST's pipeline gate is: total cells >= min_total_cells AND >= 1 per
        # group. Setting min_cells_per_group=1, min_total_cells=20 reproduces it.
        n_tot = n_ref + n_test
        if n_ref < min_cells_per_group or n_test < min_cells_per_group or n_tot < min_total_cells:
            reason = SKIP_ONE_GROUP_MISSING if (n_ref == 0 or n_test == 0) else SKIP_TOO_FEW_CELLS
            _ph_log(ct, f"skip cell type: ref={n_ref}, test={n_test}, total={n_tot} "
                        f"(min/group={min_cells_per_group}, min_total={min_total_cells}) [{reason}]")
            diag_rows.append({"celltype": ct, "comparison": comparison, "status": "skipped_celltype",
                              "skip_reason": reason, "n_ref": n_ref, "n_test": n_test,
                              "n_genes": n_total_genes, "n_tested": 0, "seconds": 0.0})
            if emit_skipped_celltypes:
                blank = [
                    {**_blank_row(g, ct, comparison, ref_label, test_label), "skip_reason": reason,
                     "n_cells": int(mask_ct.sum()), "n_ref": n_ref, "n_test": n_test}
                    for g in var_names
                ]
                writer.write_block(pd.DataFrame(blank, columns=HURDLE_OUTPUT_COLUMNS))
                total_written += len(blank)
            continue

        # ---- subset (RAM-safe sparse row slice; never densified) ----
        X_ct = X[mask_ct]
        if sp.issparse(X_ct):
            X_ct = X_ct.tocsc()
        obs_ct = obs.loc[mask_ct]

        df_ct = _process_celltype(
            ct, X_ct, var_names, cond_ct.astype(np.int8), obs_ct,
            n_total_genes=n_total_genes, comparison=comparison,
            ref_label=ref_label, test_label=test_label, covariates=covariates,
            min_positive=min_positive, detection_fallback=detection_fallback,
            logistic_engine=logistic_engine, apply_log1p=apply_log1p,
            workers=workers, chunk_genes=chunk_genes, memory_log=memory_log,
            partial_hurdle=partial_hurdle, coef_mode=coef_mode, eb_shrinkage=eb_shrinkage,
            vectorized=vectorized,
        )

        n_total_ct = len(df_ct)
        n_tested = int(df_ct["tested"].sum())
        skip_counts = (
            df_ct.loc[~df_ct["tested"].astype(bool), "skip_reason"]
            .value_counts().to_dict()
        )
        if drop_untested:
            # MAST-like compact output: write only statistically tested genes,
            # dropping all_zero / too_few_positive / no_positive_in_one_group / ...
            df_ct = df_ct[df_ct["tested"].astype(bool)]

        writer.write_block(df_ct)
        total_written += len(df_ct)

        elapsed = time.time() - t_ct
        _ph_log(ct, f"done: tested={n_tested}/{n_total_ct} written={len(df_ct)} "
                    f"skips={skip_counts} in {elapsed:.1f}s")
        diag_rows.append({"celltype": ct, "comparison": comparison, "status": "ok",
                          "skip_reason": "", "n_ref": n_ref, "n_test": n_test,
                          "n_genes": n_total_ct, "n_tested": n_tested, "n_written": len(df_ct),
                          "seconds": round(elapsed, 2), **{f"skip_{k}": v for k, v in skip_counts.items()}})

        del X_ct, obs_ct, df_ct
        gc.collect()

    writer.close()
    pd.DataFrame(diag_rows).to_csv(diag_path, index=False)
    _ph_log("setup", f"FINISHED: wrote {total_written} rows -> {out_path}")
    _ph_log("setup", f"diagnostics -> {diag_path}")
    return out_path


def run_python_hurdle_de(
    adata,
    output_dir="results",
    celltype_col="celltype",
    sample_col="Sample",
    ref_label="Reference",
    test_label="pancreas",
    min_cells_per_group=10,
    min_positive=10,
    covariates=("CDR",),
    workers=1,
    chunk_genes=500,
    output_prefix="duet",
    *,
    condition_col=None,
    detection_fallback="chisq",
    logistic_engine="statsmodels",
    assume_logged=None,
    comparison=None,
    emit_skipped_celltypes=False,
    output_format="csv",
    memory_log=True,
    coef_mode="mast_logfc",
    linear_logfc=False,
    partial_hurdle=False,
    min_total_cells=0,
    mast_compat=False,
    eb_shrinkage=None,
    output_name=None,
    vectorized=True,
    drop_untested=False,
):
    """Run the MAST-like hurdle DE for **Reference vs test** per cell type.

    Parameters
    ----------
    adata : AnnData
        ``adata.X`` is log-normalized expression (scipy sparse CSR/CSC or dense).
        Never densified in full.
    output_dir : str
        Results directory (created if missing).
    celltype_col, sample_col : str
        ``adata.obs`` columns for cell type and sample/condition source.
    ref_label, test_label : str
        Condition labels. ``ref_label`` is the model reference (coef sign is
        test-vs-reference). Condition is derived from ``sample_col`` (or
        ``condition_col`` if given) using a case-insensitive substring match.
    min_cells_per_group : int
        Minimum cells per group required to analyze a cell type.
    min_positive : int
        Minimum positive (y>0) cells required to test a gene.
    covariates : tuple[str]
        Nuisance covariates. ``"CDR"`` is computed internally; any other name is
        pulled from ``adata.obs`` (numeric z-scored, categorical one-hot).
    workers : int
        Process workers for gene-chunk parallelism. ``1`` (default) is the
        stable serial path. On Windows, ``>1`` increases RAM (no fork/COW).
    chunk_genes : int
        Genes per work unit (also the parallel granularity).
    output_prefix : str
        Output basename -> ``{prefix}_all_celltypes.{csv|parquet}`` and
        ``{prefix}_diagnostics.csv``.
    detection_fallback : {'chisq','fisher','none'}
        Fallback for the detection component when the covariate-adjusted
        logistic fit fails or shows perfect separation.
    logistic_engine : {'statsmodels','numpy_irls'}
        Engine for the detection GLM. ``numpy_irls`` is faster at scale.
    assume_logged : bool | None
        ``None`` auto-detects unlogged data (max>100) and applies log1p.
    comparison : str | None
        Value for the ``comparison`` column. Defaults to
        ``f"{ref_label}_vs_{test_label}"``.
    emit_skipped_celltypes : bool
        If True, also emit per-gene skip rows for cell types that fail the
        cell-type gate (default False: such cell types are logged and skipped).
    output_format : {'csv','parquet'}
    memory_log : bool
        Log RSS before/after each cell type if psutil is available.
    coef_mode : {'mast_logfc','mast_logfc_linear','continuous'}
        What goes in the ``coef`` column. ``mast_logfc`` (default) = the MAST
        hurdle log-fold-change (marginal mean difference, mean_expr_test -
        mean_expr_ref); this matches R MAST's ``coef`` and the downstream
        ``|logFC|>0.25`` thresholds. ``mast_logfc_linear`` = the scanpy/Wilcoxon
        style log2 fold change ``log2((expm1(mean_test)+e)/(expm1(mean_ref)+e))``,
        i.e. the same marginal effect back-transformed to the linear scale so its
        MAGNITUDE is comparable to diffxpy/DESeq2/Wilcoxon (the raw log1p-mean
        difference is compressed by Jensen's inequality). ``mast_logfc_linear`` is
        respected even under ``mast_compat``. ``continuous`` = the Gaussian-component
        condition coefficient. The continuous coefficient is always also kept
        in the ``coef_continuous`` column.
    linear_logfc : bool
        Convenience switch for the ``coef`` scale. ``False`` (DEFAULT) = the
        MAST-style COMPRESSED log-mean-difference (so DUET's ``coef`` resembles
        R MAST's); ``True`` = the scanpy/Wilcoxon-style LINEAR log2 fold change.
        Equivalent to setting ``coef_mode`` to ``mast_logfc`` / ``mast_logfc_linear``
        respectively, and overrides ``coef_mode`` when True.
    partial_hurdle : bool
        If True, test any gene with at least one estimable component (MAST-like:
        a detection-only gene with positives in just one group is still tested
        via its detection component). If False (default), require both
        components and emit the original skip reasons.
    min_total_cells : int
        Minimum total cells (both groups) to analyze a cell type. R MAST's
        pipeline uses 20.
    mast_compat : bool
        Convenience preset that reproduces R MAST's pipeline gating:
        ``min_cells_per_group=1``, ``min_total_cells=20``, ``min_positive=2``,
        ``covariates=()``, ``partial_hurdle=True``, ``coef_mode='mast_logfc'``,
        and (since MAST does this) ``eb_shrinkage=True``. These override the
        corresponding arguments.
    output_name : str | None
        Exact output filename (e.g. ``"mast_all_celltypes.csv"``). Overrides the
        ``{prefix}_all_celltypes.{ext}`` convention so DUET can be a drop-in
        replacement for the pipeline's MAST output; diagnostics are written as
        ``{stem}_diagnostics.csv`` next to it.
    eb_shrinkage : bool | None
        Empirical-Bayes (limma/Smyth) moderation of the continuous-component
        variance: shrink per-gene residual variances toward a pooled prior and
        use a moderated t-test for the condition effect. This is the main thing
        R MAST does that a plain per-gene OLS does not, and it brings the FDR
        calibration close to MAST. ``None`` (default) means auto: on under
        ``mast_compat``, off otherwise.

    Returns
    -------
    str
        Path to ``{prefix}_all_celltypes.{csv|parquet}``.
    """
    if linear_logfc:                       # boolean switch wins over coef_mode
        coef_mode = "mast_logfc_linear"
    if mast_compat:
        min_cells_per_group = 1
        min_total_cells = max(min_total_cells, 20)
        min_positive = min(min_positive, 2)
        covariates = ()
        partial_hurdle = True
        if coef_mode != "mast_logfc_linear":  # respect an explicit linear-logFC request
            coef_mode = "mast_logfc"
        _ph_log("setup", "mast_compat=True: gate>=1/group & >=20 total, no covariates, "
                         f"partial hurdle, coef_mode={coef_mode}, EB shrinkage.")

    # Resolve the eb_shrinkage sentinel: auto-on under mast_compat.
    eb_shrinkage = bool(eb_shrinkage) if eb_shrinkage is not None else bool(mast_compat)

    if comparison is None:
        comparison = f"{ref_label}_vs_{test_label}"

    cond_all = _derive_condition(adata.obs, sample_col, ref_label, test_label, condition_col)
    n_ref = int(np.sum(cond_all == 0))
    n_test = int(np.sum(cond_all == 1))
    _ph_log("setup", f"global condition counts: {ref_label}={n_ref}, {test_label}={n_test}")
    if n_ref == 0 or n_test == 0:
        _ph_log("setup", "ABORT: one global condition group is empty; nothing to compare.")
        # Still create an (empty) output so downstream code finds the file.
        os.makedirs(output_dir, exist_ok=True)
        if output_name:
            out_path = os.path.join(output_dir, output_name)
        else:
            ext = "parquet" if str(output_format).lower() == "parquet" else "csv"
            out_path = os.path.join(output_dir, f"{output_prefix}_all_celltypes.{ext}")
        pd.DataFrame(columns=HURDLE_OUTPUT_COLUMNS).to_csv(out_path, index=False)
        return out_path

    return _run_hurdle_core(
        adata, cond_all=cond_all, celltype_col=celltype_col, output_dir=output_dir,
        output_prefix=output_prefix, comparison=comparison, ref_label=ref_label,
        test_label=test_label, min_cells_per_group=min_cells_per_group,
        min_positive=min_positive, covariates=covariates, workers=workers,
        chunk_genes=chunk_genes, detection_fallback=detection_fallback,
        logistic_engine=logistic_engine, assume_logged=assume_logged,
        emit_skipped_celltypes=emit_skipped_celltypes, output_format=output_format,
        memory_log=memory_log, min_total_cells=min_total_cells,
        partial_hurdle=partial_hurdle, coef_mode=coef_mode, eb_shrinkage=eb_shrinkage,
        output_name=output_name, vectorized=vectorized, drop_untested=drop_untested,
    )


def run_python_hurdle_cnv(
    adata,
    output_dir="results",
    celltype_col="celltype",
    cnv_col="cnv_score",
    cnv_threshold=0.05,
    ref_label="CNVLow",
    test_label="CNVHigh",
    min_cells_per_group=10,
    min_positive=10,
    covariates=("CDR",),
    workers=1,
    chunk_genes=500,
    output_prefix="python_hurdle_cnv",
    *,
    detection_fallback="chisq",
    logistic_engine="statsmodels",
    assume_logged=None,
    emit_skipped_celltypes=False,
    output_format="csv",
    memory_log=True,
    coef_mode="mast_logfc",
    linear_logfc=False,
    partial_hurdle=False,
    min_total_cells=0,
    mast_compat=False,
    eb_shrinkage=None,
    output_name=None,
    vectorized=True,
    drop_untested=False,
):
    """CNVHigh vs CNVLow hurdle DE per cell type (requirement #7).

    Condition is derived from ``adata.obs[cnv_col]``: a cell is *test* (CNVHigh)
    iff its CNV score ``>= cnv_threshold``, else *reference* (CNVLow). Reuses the
    same hurdle engine as :func:`run_python_hurdle_de`; see that function for the
    ``coef_mode`` / ``partial_hurdle`` / ``min_total_cells`` / ``mast_compat``
    parameter semantics.
    """
    if cnv_col not in adata.obs.columns:
        raise KeyError(f"'{cnv_col}' not found in adata.obs (run inferCNV first)")

    if linear_logfc:
        coef_mode = "mast_logfc_linear"
    if mast_compat:
        min_cells_per_group = 1
        min_total_cells = max(min_total_cells, 20)
        min_positive = min(min_positive, 2)
        covariates = ()
        partial_hurdle = True
        if coef_mode != "mast_logfc_linear":  # respect an explicit linear-logFC request
            coef_mode = "mast_logfc"
    eb_shrinkage = bool(eb_shrinkage) if eb_shrinkage is not None else bool(mast_compat)

    cnv = pd.to_numeric(adata.obs[cnv_col], errors="coerce").to_numpy()
    cond_all = np.where(cnv >= float(cnv_threshold), 1, 0).astype(np.int8)
    cond_all[np.isnan(cnv)] = 0  # undefined CNV -> treat as low/reference
    comparison = f"{test_label}_vs_{ref_label}_thr{cnv_threshold}"

    n_ref = int(np.sum(cond_all == 0))
    n_test = int(np.sum(cond_all == 1))
    _ph_log("setup", f"CNV split @ {cnv_threshold}: {ref_label}={n_ref}, {test_label}={n_test}")

    return _run_hurdle_core(
        adata, cond_all=cond_all, celltype_col=celltype_col, output_dir=output_dir,
        output_prefix=output_prefix, comparison=comparison, ref_label=ref_label,
        test_label=test_label, min_cells_per_group=min_cells_per_group,
        min_positive=min_positive, covariates=covariates, workers=workers,
        chunk_genes=chunk_genes, detection_fallback=detection_fallback,
        logistic_engine=logistic_engine, assume_logged=assume_logged,
        emit_skipped_celltypes=emit_skipped_celltypes, output_format=output_format,
        memory_log=memory_log, min_total_cells=min_total_cells,
        partial_hurdle=partial_hurdle, coef_mode=coef_mode, eb_shrinkage=eb_shrinkage,
        output_name=output_name, vectorized=vectorized, drop_untested=drop_untested,
    )


# DUET public entry points. `run_duet` is the preferred name; run_scpyde /
# run_python_hurdle_* are kept as backward-compatible aliases (the method was
# previously branded "scPyDE", which collided with SCDE -- an unrelated Bayesian
# dropout method -- and before that "python_hurdle").
run_duet = run_python_hurdle_de
run_duet_cnv = run_python_hurdle_cnv
run_scpyde = run_python_hurdle_de          # deprecated
run_scpyde_cnv = run_python_hurdle_cnv     # deprecated


# --------------------------------------------------------------------------- #
# Standalone CLI (useful for testing on a saved .h5ad without the full pipeline)
# --------------------------------------------------------------------------- #
def _main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Run Python hurdle DE on an .h5ad file.")
    ap.add_argument("h5ad", help="Path to an AnnData .h5ad with log-normalized X")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--celltype-col", default="celltype")
    ap.add_argument("--sample-col", default="Sample")
    ap.add_argument("--ref-label", default="Reference")
    ap.add_argument("--test-label", default="pancreas")
    ap.add_argument("--min-cells-per-group", type=int, default=10)
    ap.add_argument("--min-positive", type=int, default=10)
    ap.add_argument("--covariates", default="CDR", help="comma-separated; e.g. 'CDR,Sample'")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk-genes", type=int, default=500)
    ap.add_argument("--output-prefix", default="python_hurdle")
    ap.add_argument("--detection-fallback", default="chisq", choices=["chisq", "fisher", "none"])
    ap.add_argument("--logistic-engine", default="statsmodels", choices=["statsmodels", "numpy_irls"])
    ap.add_argument("--output-format", default="csv", choices=["csv", "parquet"])
    ap.add_argument("--mode", default="reference", choices=["reference", "cnv"])
    ap.add_argument("--cnv-col", default="cnv_score")
    ap.add_argument("--cnv-threshold", type=float, default=0.05)
    ap.add_argument("--coef-mode", default="mast_logfc",
                    choices=["mast_logfc", "mast_logfc_linear", "continuous"])
    ap.add_argument("--linear-logfc", action="store_true",
                    help="Report the LINEAR (scanpy/Wilcoxon) log2 fold change instead of the "
                         "default COMPRESSED MAST-style log-mean-difference. "
                         "(= --coef-mode mast_logfc_linear; default off -> resembles R MAST.)")
    ap.add_argument("--partial-hurdle", action="store_true",
                    help="Test any gene with an estimable component (MAST-like).")
    ap.add_argument("--min-total-cells", type=int, default=0)
    ap.add_argument("--mast-compat", action="store_true",
                    help="Preset matching R MAST gating (>=1/group, >=20 total, no covariates, "
                         "partial hurdle, MAST logFC coef, EB shrinkage).")
    ap.add_argument("--eb-shrinkage", action=argparse.BooleanOptionalAction, default=None,
                    help="Empirical-Bayes variance moderation of the continuous component "
                         "(--eb-shrinkage / --no-eb-shrinkage). Default: auto (on with --mast-compat).")
    ap.add_argument("--vectorized", action=argparse.BooleanOptionalAction, default=True,
                    help="Use the vectorized two-group fast path when there are no covariates "
                         "(--vectorized / --no-vectorized). Default: on.")
    ap.add_argument("--drop-untested", action="store_true",
                    help="Write only statistically tested genes (drop all_zero / too_few_positive "
                         "/ no_positive_in_one_group rows). Default: keep all post-filter genes.")
    args = ap.parse_args(argv)

    import anndata as ad

    adata = ad.read_h5ad(args.h5ad)
    covariates = tuple(c.strip() for c in args.covariates.split(",") if c.strip())
    common = dict(
        output_dir=args.output_dir, celltype_col=args.celltype_col,
        min_cells_per_group=args.min_cells_per_group, min_positive=args.min_positive,
        covariates=covariates, workers=args.workers, chunk_genes=args.chunk_genes,
        output_prefix=args.output_prefix, detection_fallback=args.detection_fallback,
        logistic_engine=args.logistic_engine, output_format=args.output_format,
        coef_mode=args.coef_mode, linear_logfc=args.linear_logfc,
        partial_hurdle=args.partial_hurdle,
        min_total_cells=args.min_total_cells, mast_compat=args.mast_compat,
        eb_shrinkage=args.eb_shrinkage, vectorized=args.vectorized,
        drop_untested=args.drop_untested,
    )

    if args.mode == "cnv":
        path = run_python_hurdle_cnv(
            adata, cnv_col=args.cnv_col, cnv_threshold=args.cnv_threshold, **common,
        )
    else:
        path = run_python_hurdle_de(
            adata, sample_col=args.sample_col, ref_label=args.ref_label,
            test_label=args.test_label, **common,
        )
    print(f"[PY-HURDLE] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
