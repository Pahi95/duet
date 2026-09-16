#!/usr/bin/env python
"""
Tests for numerical edge cases and input validation.

1. The moderated-t tail is carried in log space. The empirical-Bayes step used to
   clip the two-sided t p-value at 1e-300, which capped the continuous statistic at
   chi2.isf(1e-300, 1) = 1373.87 and tied every gene beyond it. These tests pin the
   log-space tail against SciPy where SciPy works and against mpmath past the floor.

2. Unusual inputs: all-zero, near-empty, zero-variance, extremely unbalanced and
   separated genes; CSR, CSC and dense matrices; float32 and float64; gene order.
   None of these may crash, and none may change a result that should not change.

    pytest tests/ -q
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2, t as student_t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from duet.core import (_log_betainc_small, _log_two_sided_t_sf,    # noqa: E402
                       _moderated_t_to_chi2)

OLD_CAP = float(chi2.isf(1e-300, 1))          # 1373.87, the statistic's old ceiling


def _mp_log_two_sided_t_sf(t, df, dps=60):
    """Reference: log(2 * P(T_df > |t|)) in mpmath, by one of two independent routes.

    Up to df = 1e4: the regularised incomplete beta I_x(df/2, 1/2), x = df/(df+t^2).
    Above that mpmath's betainc stalls, and the density is integrated instead:
    2 f(t) * integral_0^inf exp(log f(t+s) - log f(t)) ds. (The integral is not used
    for small df: with t ~ 1e110 its heavy tail defeats the quadrature, while the
    closed form x^a / (a B(a, b)) confirms the continued fraction to 1e-12.)
    """
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = dps
    t, nu = mp.mpf(abs(t)), mp.mpf(df)
    if df <= 1e4:
        x = nu / (nu + t * t)
        return float(mp.log(mp.betainc(nu / 2, mp.mpf(1) / 2, 0, x, regularized=True)))
    const = mp.loggamma((nu + 1) / 2) - mp.loggamma(nu / 2) - mp.log(nu * mp.pi) / 2

    def logpdf(u):
        return const - (nu + 1) / 2 * mp.log1p(u * u / nu)

    ref = logpdf(t)
    pts = sorted({mp.mpf(0), 1 / t, 10 / t, 100 / t, t / 10, t, 10 * t})
    integral = mp.quad(lambda s: mp.exp(logpdf(t + s) - ref), pts + [mp.inf])
    return float(mp.log(2) + ref + mp.log(integral))


# --------------------------------------------------------------------------
# 1. log-space moderated-t tail
# --------------------------------------------------------------------------

@pytest.mark.parametrize("df", [3.0, 30.0, 300.0, 5000.0, 1e6])
def test_log_betainc_matches_scipy_where_scipy_works(df):
    """The continued fraction is checked against SciPy on the range both can compute."""
    targets = np.linspace(-20.0, -290.0, 60)                  # log10 of the two-sided p
    t = student_t.isf(10.0 ** targets / 2.0, df)
    with np.errstate(divide="ignore"):
        want = np.log(2.0 * student_t.sf(t, df))
    # compare only where SciPy is exact: once its tail is subnormal (< ~2e-308) it
    # keeps just a few significant digits and is the less accurate of the two
    ok = np.isfinite(want) & (want > math.log(1e-300))
    assert ok.sum() > 40
    log_t2 = 2.0 * np.log(t[ok])
    log_den = np.logaddexp(math.log(df), log_t2)
    got = _log_betainc_small(np.full(ok.sum(), df / 2.0), 0.5, math.log(df) - log_den, log_t2 - log_den)
    assert np.allclose(got, want[ok], rtol=1e-10, atol=0)


@pytest.mark.parametrize("df", [3.0, 10.0, 30.0, 100.0, 1000.0, 1e4, 1e5, 1e6])
def test_log_t_tail_matches_mpmath_past_the_floor(df):
    """Past 1e-308 SciPy returns 0; the log-space tail must still be exact."""
    # t values whose two-sided tail spans roughly 1e-310 .. 1e-20000
    if df <= 3:                               # heavy tail: p ~ t^-3, so t must be astronomical
        ts = [1e110, 1e150, 1e200]
    else:
        base = float(student_t.isf(1e-300, df))
        ts = list(base * np.array([1.05, 1.5, 3.0, 10.0, 50.0]))
    got = _log_two_sided_t_sf(np.array(ts), df)
    for g, tt in zip(got, ts):
        w = _mp_log_two_sided_t_sf(tt, df)
        assert np.isfinite(g)
        assert abs(g - w) <= 1e-10 * abs(w), f"df={df} t={tt}: {g} vs {w}"


def test_statistic_is_unchanged_where_scipy_works():
    """Above the old floor nothing moves: same p, bit-identical chi-square."""
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 30, 2000)
    df = rng.uniform(5, 1e5, 2000)
    p, stat, _ = _moderated_t_to_chi2(t, df)
    p_old = np.clip(2.0 * student_t.sf(t, df), 1e-300, 1.0)
    assert np.array_equal(p, p_old)
    assert np.array_equal(stat, chi2.isf(p_old, 1))


def test_statistic_keeps_growing_past_the_old_cap():
    df = 2000.0
    t = np.geomspace(float(student_t.isf(1e-290, df)), 5e3, 400)
    p, stat, log_p = _moderated_t_to_chi2(t, df)
    assert np.all(np.diff(stat) > 0), "must stay strictly ordered past the floor"
    assert stat.max() > 10 * OLD_CAP
    assert np.all(np.isfinite(log_p)) and np.all(np.diff(log_p) < 0)
    # the chi-square has the same upper tail as the two-sided t: log sf_chi2_1(stat) == log_p
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 80
    for s, lp in list(zip(stat, log_p))[::57]:
        ref = float(mp.log(mp.erfc(mp.sqrt(mp.mpf(s) / 2))))     # chi2_1 sf = erfc(sqrt(x/2))
        assert abs(ref - lp) <= 1e-9 * abs(lp)


def test_no_step_at_the_switchover():
    df = 800.0
    t = np.linspace(float(student_t.isf(1e-280, df)), float(student_t.isf(1e-300, df)) * 1.02, 3001)
    _, stat, _ = _moderated_t_to_chi2(t, df)
    steps = np.diff(stat)
    assert np.all(steps > 0)
    assert steps.max() < 5 * np.median(steps)


# --------------------------------------------------------------------------
# helpers for end-to-end runs
# --------------------------------------------------------------------------

def _adata(X, cond, genes=None):
    import anndata as ad
    import pandas as pd
    n, g = X.shape
    obs = pd.DataFrame({"celltype": "T", "Sample": cond}, index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=genes if genes is not None else [f"g{j}" for j in range(g)])
    return ad.AnnData(X=X, obs=obs, var=var)


def _run(adata, tmp, **kw):
    import pandas as pd
    from duet import run_duet
    out = run_duet(adata, output_dir=str(tmp), celltype_col="celltype",
                   sample_col="Sample", ref_label="ctrl", test_label="stim",
                   memory_log=False, output_name="t.csv", **kw)
    return pd.read_csv(out).set_index("gene")


def _mixed(n_cells=800, seed=1):
    """Ordinary genes plus every awkward case the engine has to survive."""
    rng = np.random.default_rng(seed)
    half = n_cells // 2
    cond = np.array(["ctrl"] * half + ["stim"] * (n_cells - half))
    cols = {}
    for g in range(20):                                           # ordinary, some with signal
        p = (0.3, 0.6) if g < 5 else (0.4, 0.4)
        mu = (1.0, 1.8) if 5 <= g < 10 else (1.2, 1.2)
        v = np.zeros(n_cells)
        for lo, hi, pr, m in ((0, half, p[0], mu[0]), (half, n_cells, p[1], mu[1])):
            v[lo:hi] = (rng.random(hi - lo) < pr) * rng.lognormal(m, 0.4, hi - lo)
        cols[f"ord{g}"] = v
    cols["all_zero"] = np.zeros(n_cells)
    one = np.zeros(n_cells); one[3] = 2.0
    cols["one_positive"] = one
    const = np.where(rng.random(n_cells) < 0.5, 1.0, 0.0)       # zero variance among positives
    cols["zero_variance"] = const
    sep = np.zeros(n_cells); sep[half:] = rng.lognormal(1.0, 0.3, n_cells - half)
    cols["separated"] = sep                                       # detected only in stim
    everywhere = rng.lognormal(1.0, 0.3, n_cells)                 # detected in every cell
    cols["always_detected"] = everywhere
    X = np.column_stack(list(cols.values()))
    return X, cond, list(cols)


# --------------------------------------------------------------------------
# 2. edge cases
# --------------------------------------------------------------------------

@pytest.mark.parametrize("vectorized", [True, False])
def test_awkward_genes_do_not_crash_or_leak_nan(tmp_path, vectorized):
    import scipy.sparse as sp
    X, cond, genes = _mixed()
    d = _run(_adata(sp.csr_matrix(X), cond, genes), tmp_path, mast_compat=True, vectorized=vectorized)
    assert len(d) == len(genes)
    assert not bool(d.loc["all_zero", "tested"])
    t = d[d.tested == True]                                           # noqa: E712
    assert t.pvalue.between(0, 1).all()
    assert np.isfinite(t.stat_hurdle).all() and np.isfinite(t.neglog10p).all()
    assert bool(d.loc["separated", "detection_separated"])
    assert bool(d.loc["separated", "tested"]) and d.loc["separated", "neglog10p"] > 10


def test_extremely_unbalanced_groups(tmp_path):
    import scipy.sparse as sp
    rng = np.random.default_rng(4)
    n_ref, n_test, g = 990, 10, 30
    X = (rng.random((n_ref + n_test, g)) < 0.5) * rng.lognormal(1.0, 0.5, (n_ref + n_test, g))
    cond = np.array(["ctrl"] * n_ref + ["stim"] * n_test)
    for vec in (True, False):
        d = _run(_adata(sp.csr_matrix(X), cond), tmp_path / str(vec), mast_compat=True, vectorized=vec)
        t = d[d.tested == True]                                       # noqa: E712
        assert len(t) > 20
        assert t.pvalue.between(0, 1).all()
        assert (t.fdr < 0.05).sum() <= 2, "null data with a 10-cell arm must not light up"


def test_continuous_statistic_is_no_longer_capped(tmp_path):
    """Huge, graded effects on many cells: the old code gave all of them 1373.87."""
    import scipy.sparse as sp
    rng = np.random.default_rng(5)
    n = 6000
    cond = np.array(["ctrl"] * (n // 2) + ["stim"] * (n // 2))
    shifts = [1.0, 1.5, 2.0, 2.5]
    cols = []
    for s in shifts:
        v = np.exp(rng.normal(1.0, 0.1, n))
        v[n // 2:] *= math.exp(s)
        cols.append(np.log1p(v))
    X = sp.csr_matrix(np.column_stack(cols + [np.log1p(np.exp(rng.normal(1, 0.1, n))) for _ in range(12)]))
    for vec in (True, False):
        d = _run(_adata(X, cond), tmp_path / str(vec), mast_compat=True, vectorized=vec)
        sc = d.loc[[f"g{j}" for j in range(4)], "stat_continuous"].to_numpy()
        assert (sc > OLD_CAP).all(), sc
        assert np.all(np.diff(sc) > 0), "larger shift must give a larger statistic"


# --------------------------------------------------------------------------
# 3. input representation must not change the answer
# --------------------------------------------------------------------------

@pytest.mark.parametrize("vectorized", [True, False])
def test_csr_csc_and_dense_agree(tmp_path, vectorized):
    import scipy.sparse as sp
    X, cond, genes = _mixed(seed=2)
    runs = {}
    for name, M in (("csr", sp.csr_matrix(X)), ("csc", sp.csc_matrix(X)), ("dense", X.copy())):
        runs[name] = _run(_adata(M, cond, genes), tmp_path / name, mast_compat=True, vectorized=vectorized)
    ref = runs["csr"]
    m = ref.tested == True                                            # noqa: E712
    for name in ("csc", "dense"):
        d = runs[name].reindex(ref.index)
        assert (d.tested == ref.tested).all()
        assert np.allclose(d.loc[m, "stat_hurdle"], ref.loc[m, "stat_hurdle"], rtol=1e-10, atol=1e-10)
        assert np.allclose(d.loc[m, "coef"], ref.loc[m, "coef"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("vectorized", [True, False])
def test_float32_and_float64_agree(tmp_path, vectorized):
    import scipy.sparse as sp
    X, cond, genes = _mixed(seed=3)
    d64 = _run(_adata(sp.csr_matrix(X.astype(np.float64)), cond, genes), tmp_path / "64",
               mast_compat=True, vectorized=vectorized)
    d32 = _run(_adata(sp.csr_matrix(X.astype(np.float32)), cond, genes), tmp_path / "32",
               mast_compat=True, vectorized=vectorized).reindex(d64.index)
    m = d64.tested == True                                            # noqa: E712
    assert (d32.tested == d64.tested).all()
    # float32 storage perturbs the inputs at ~1e-7 relative; results may move that much, no more
    assert np.allclose(d32.loc[m, "neglog10p"], d64.loc[m, "neglog10p"], rtol=1e-4, atol=1e-4)


def test_pseudobulk_does_not_depend_on_cell_order():
    """PyDESeq2's paired fit depends on sample order, so the aggregation must fix it."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    from duet.pseudobulk import aggregate_pseudobulk
    rng = np.random.default_rng(8)
    n = 600
    C = rng.poisson(2.0, (n, 30))
    obs = pd.DataFrame({"donor": np.repeat([f"d{i}" for i in range(6)], 100),
                        "Sample": np.tile(np.repeat(["ctrl", "stim"], 50), 6)},
                       index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X=sp.csr_matrix(np.log1p(C).astype(float)), obs=obs)
    a.layers["counts"] = sp.csr_matrix(C)
    kw = dict(donor_col="donor", condition_col="Sample", ref_label="ctrl", test_label="stim")
    pb1, m1 = aggregate_pseudobulk(a, **kw)
    pb2, m2 = aggregate_pseudobulk(a[rng.permutation(n)].copy(), **kw)
    assert list(pb1.index) == list(pb2.index)
    assert np.array_equal(pb1.to_numpy(), pb2.to_numpy())
    assert m1.equals(m2)


def test_combined_rule_needs_the_two_arms_to_agree_on_direction():
    """The manuscript's rule: pseudobulk decides, but only when the hurdle points the
    same way. A gene the two arms call in opposite directions is not a discovery."""
    import pandas as pd
    from duet.pseudobulk import (CALL_DIRECTION_CONFLICT, CALL_NS, CALL_RANKED_ONLY,
                                 CALL_SIGNIFICANT, combine_calls)
    duet = pd.DataFrame({"gene": list("abcd"), "celltype": "T", "fdr": [0.01, 0.01, 0.5, 0.01],
                         "coef": [1.0, -1.0, 1.0, 1.0]})
    pb = pd.DataFrame({"gene": list("abcd"), "celltype": "T", "pb_padj": [0.01, 0.01, 0.01, 0.9],
                       "pb_log2FC": [2.0, 2.0, 2.0, 2.0], "pb_pvalue": 0.001, "pb_tested": True})
    got = combine_calls(duet, pb).set_index("gene")["call"]
    assert got["a"] == CALL_SIGNIFICANT          # both arms, same direction
    assert got["b"] == CALL_DIRECTION_CONFLICT   # pseudobulk up, hurdle down
    assert got["c"] == CALL_SIGNIFICANT          # pseudobulk decides; the hurdle need not agree
    assert got["d"] == CALL_RANKED_ONLY          # hurdle only
    relaxed = combine_calls(duet, pb, require_same_direction=False).set_index("gene")["call"]
    assert relaxed["b"] == CALL_SIGNIFICANT
    assert set(got) <= {CALL_SIGNIFICANT, CALL_DIRECTION_CONFLICT, CALL_RANKED_ONLY, CALL_NS}


def test_run_duet_pseudobulk_end_to_end(tmp_path):
    """Both arms from one call: a hurdle ranking plus sample-level calls."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    from duet import run_duet_pseudobulk
    rng = np.random.default_rng(12)
    n_donors, per = 6, 120
    donors = np.repeat([f"d{i}" for i in range(n_donors)], per)
    cond = np.where(np.isin(donors, [f"d{i}" for i in range(n_donors // 2)]), "ctrl", "stim")
    n, g = len(donors), 60
    base = rng.gamma(2.0, 2.0, g)
    C = rng.poisson(base[None, :] * np.where(cond[:, None] == "stim", 1.0, 1.0))
    C[:, :10] = rng.poisson(base[None, :10] * np.where(cond[:, None] == "stim", 6.0, 1.0))  # real signal
    X = sp.csr_matrix(np.log1p(C / np.maximum(C.sum(1, keepdims=True), 1) * 1e4))
    a = ad.AnnData(X=X, obs=pd.DataFrame({"celltype": "T", "Sample": cond, "donor": donors},
                                         index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))
    a.layers["counts"] = sp.csr_matrix(C)
    out = run_duet_pseudobulk(a, output_dir=str(tmp_path), celltype_col="celltype", condition_col="Sample",
                              ref_label="ctrl", test_label="stim", donor_col="donor", n_cpus=1)
    d = pd.read_csv(out)
    assert {"neglog10p", "pb_padj", "call"} <= set(d.columns)
    planted = {f"g{j}" for j in range(10)}
    sig = set(d.loc[d["call"] == "significant", "gene"])
    assert len(sig & planted) >= 8, f"should recover the planted genes, got {sorted(sig & planted)}"
    # a couple of false positives are ordinary at FDR 0.05 over 50 null genes; a flood is not
    assert len(sig - planted) <= 3, f"too many false positives: {sorted(sig - planted)}"


def test_gene_order_does_not_matter(tmp_path):
    import scipy.sparse as sp
    X, cond, genes = _mixed(seed=6)
    perm = np.random.default_rng(0).permutation(len(genes))
    a = _run(_adata(sp.csr_matrix(X), cond, genes), tmp_path / "a", mast_compat=True)
    b = _run(_adata(sp.csr_matrix(X[:, perm]), cond, [genes[i] for i in perm]), tmp_path / "b",
             mast_compat=True).reindex(a.index)
    m = a.tested == True                                              # noqa: E712
    assert np.allclose(a.loc[m, "stat_hurdle"], b.loc[m, "stat_hurdle"], rtol=1e-12, atol=1e-12)
    assert np.allclose(a.loc[m, "fdr"], b.loc[m, "fdr"], rtol=1e-12, atol=1e-12)
