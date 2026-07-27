#!/usr/bin/env python
"""
Tests for DUET.

These are correctness tests, not smoke tests. Each one pins a property that a
silent change would break and that the benchmarks would not catch: the χ² tail
expansion against SciPy, the equivalence of the vectorised and general code
paths, the underflow behaviour that motivates `neglog10p`, and the invariants the
manuscript's claims rest on.

    pip install pytest
    pytest tests/ -q
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from duet.core import _neglog10_chi2_sf, _bh_fdr, _logit_irls   # noqa: E402


# --------------------------------------------------------------------------
# the underflow-safe tail -- the paper's numerical claim
# --------------------------------------------------------------------------

@pytest.mark.parametrize("df", [1, 2, 3, 5])
def test_neglog10p_matches_scipy_where_scipy_works(df):
    """Below the float64 floor SciPy is exact; DUET must agree with it there."""
    stat = np.linspace(0.5, 1300.0, 400)
    got = _neglog10_chi2_sf(stat, df)
    want = -chi2.logsf(stat, df) / math.log(10.0)
    ok = np.isfinite(want)
    assert ok.sum() > 100, "test needs a range SciPy can still compute"
    assert np.allclose(got[ok], want[ok], rtol=0, atol=1e-9)


def test_neglog10p_survives_where_the_pvalue_underflows():
    """The whole point: chi2.sf is 0.0 here, so a p-value column cannot rank."""
    stat = np.array([1500.0, 2000.0, 5000.0, 10000.0])
    assert np.all(chi2.sf(stat, 2) == 0.0), "premise: these underflow"
    got = _neglog10_chi2_sf(stat, 2)
    assert np.all(np.isfinite(got))
    assert np.all(np.diff(got) > 0), "must stay strictly ordered past the floor"
    # for df=2 the tail is exact: sf = exp(-x/2)
    want = (stat / 2.0) / math.log(10.0)
    assert np.allclose(got, want, rtol=1e-12)


def test_neglog10p_is_continuous_across_the_switchover():
    """No step where the code stops trusting SciPy and uses the expansion."""
    stat = np.linspace(1300.0, 1500.0, 2001)
    got = _neglog10_chi2_sf(stat, 2)
    steps = np.diff(got)
    assert np.all(steps > 0)
    # a discontinuity would show up as one step far larger than its neighbours
    assert steps.max() < 5 * np.median(steps)


def test_neglog10p_rejects_invalid_input():
    got = _neglog10_chi2_sf(np.array([-1.0, np.nan, np.inf, 0.0]), 2)
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert np.isclose(got[3], 0.0)


# --------------------------------------------------------------------------
# Benjamini-Hochberg
# --------------------------------------------------------------------------

def test_bh_matches_a_known_result():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    got = _bh_fdr(p)
    want = np.minimum.accumulate(
        (p * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    assert np.allclose(got, np.clip(want, 0, 1))


def test_bh_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    q = _bh_fdr(p)
    assert np.all((q >= 0) & (q <= 1))
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12), "BH output must not decrease with p"


def test_bh_handles_nan_without_shifting_the_rest():
    p = np.array([0.01, np.nan, 0.02, 0.5])
    q = _bh_fdr(p)
    assert np.isnan(q[1])
    assert np.all(np.isfinite(q[[0, 2, 3]]))


# --------------------------------------------------------------------------
# the logistic engine -- default changed to numpy_irls, so pin its behaviour
# --------------------------------------------------------------------------

def test_irls_recovers_a_known_coefficient():
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    true = np.array([-0.4, 1.3])
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ true)))).astype(float)
    fit = _logit_irls(X, y)
    assert fit is not None and fit["converged"]
    assert np.allclose(fit["beta"], true, atol=0.15)


def test_irls_flags_perfect_separation():
    """Separation has no MLE; the fit must say so rather than return a number."""
    n = 200
    g = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    X = np.column_stack([np.ones(n), g])
    fit = _logit_irls(X, g.copy())        # outcome == predictor
    assert fit is None or fit["separated"], "separation must be flagged"


# --------------------------------------------------------------------------
# end-to-end: the two code paths must agree, and the output contract must hold
# --------------------------------------------------------------------------

def _toy(n_cells=600, n_genes=40, seed=0):
    """Small AnnData with real signal in both hurdle components."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    half = n_cells // 2
    cond = np.array(["ctrl"] * half + ["stim"] * (n_cells - half))
    X = np.zeros((n_cells, n_genes))
    for g in range(n_genes):
        p_ctrl = 0.2 if g < 10 else 0.5          # detection-rate change
        p_stim = 0.8 if g < 10 else 0.5
        mu_ctrl, mu_stim = (1.0, 2.5) if 10 <= g < 20 else (1.5, 1.5)
        for lo, hi, pr, mu in ((0, half, p_ctrl, mu_ctrl),
                               (half, n_cells, p_stim, mu_stim)):
            det = rng.random(hi - lo) < pr
            X[lo:hi, g] = det * rng.lognormal(mu, 0.5, hi - lo)
    obs = pd.DataFrame({"celltype": "T", "Sample": cond},
                       index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(n_genes)])
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)


@pytest.fixture(scope="module")
def toy():
    return _toy()


def _run(adata, tmp, **kw):
    import pandas as pd
    from duet import run_duet
    out = run_duet(adata, output_dir=str(tmp), celltype_col="celltype",
                   sample_col="Sample", ref_label="ctrl", test_label="stim",
                   memory_log=False, output_name="t.csv", **kw)
    return pd.read_csv(out)


def test_output_contract(toy, tmp_path):
    d = _run(toy, tmp_path)
    for col in ("gene", "celltype", "pvalue", "fdr", "neglog10p", "tested",
                "detect_rate_ref", "detect_rate_test"):
        assert col in d.columns, f"missing promised column {col}"
    assert len(d) == toy.n_vars, "one row per gene, tested or not"
    t = d[d.tested == True]                                        # noqa: E712
    assert (t.fdr >= t.pvalue - 1e-12).all(), "FDR cannot be below its p-value"
    assert ((t.neglog10p >= 0) | t.neglog10p.isna()).all()


def test_vectorised_path_is_actually_taken(toy, tmp_path, monkeypatch):
    """`vectorized=True` is NOT sufficient: the fast path applies only when there
    are no covariates, and the default is covariates=("CDR",). Without
    mast_compat=True the run silently uses the general path, which made the
    equivalence test below compare the general path against itself."""
    import duet.core as core
    seen = []
    orig = core._fit_celltype_vectorized
    monkeypatch.setattr(core, "_fit_celltype_vectorized",
                        lambda *a, **k: (seen.append(1), orig(*a, **k))[1])
    _run(toy, tmp_path / "novec", vectorized=True)
    assert seen == [], "fast path should be blocked by the default CDR covariate"
    _run(toy, tmp_path / "vec", vectorized=True, mast_compat=True)
    assert len(seen) == 1, "fast path was not entered under mast_compat=True"


def test_vectorised_and_general_paths_agree(toy, tmp_path):
    """The vectorised two-group path is an optimisation, not a different test.
    mast_compat=True clears the covariates, which is what makes it eligible."""
    fast = _run(toy, tmp_path / "fast", vectorized=True,
                mast_compat=True).set_index("gene")
    slow = _run(toy, tmp_path / "slow", vectorized=False,
                mast_compat=True).set_index("gene")
    slow = slow.reindex(fast.index)
    m = (fast.tested == True) & (slow.tested == True)               # noqa: E712
    assert m.sum() > 20, "need a reasonable number of tested genes"
    assert np.allclose(fast.loc[m, "stat_hurdle"], slow.loc[m, "stat_hurdle"],
                       rtol=1e-6, atol=1e-6)
    assert np.allclose(fast.loc[m, "neglog10p"], slow.loc[m, "neglog10p"],
                       rtol=1e-6, atol=1e-6)


def test_neglog10p_ranking_agrees_with_pvalue_when_nothing_underflows(toy, tmp_path):
    """On small data the two columns must rank identically -- they only diverge
    once the p-value hits the floor, which is the paper's point."""
    from scipy.stats import spearmanr
    d = _run(toy, tmp_path)
    t = d[(d.tested == True) & (d.pvalue > 0)]                      # noqa: E712
    rho = spearmanr(-np.log10(t.pvalue), t.neglog10p).statistic
    assert rho > 0.999


def test_detects_a_pure_detection_rate_change(toy, tmp_path):
    """Genes 0-9 differ ONLY in detection rate (0.2 vs 0.8). A mean-based test
    would miss them; the hurdle is why DUET exists."""
    d = _run(toy, tmp_path).set_index("gene")
    det = [f"g{i}" for i in range(10)]
    called = d.loc[det, "fdr"] < 0.05
    assert called.sum() >= 8, f"hurdle missed detection-rate genes: {called.sum()}/10"


def test_null_data_is_calibrated(tmp_path):
    """No signal anywhere: few genes should be called at FDR < 0.05."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    rng = np.random.default_rng(7)
    n, g = 800, 300
    X = (rng.random((n, g)) < 0.4) * rng.lognormal(1.5, 0.5, (n, g))
    obs = pd.DataFrame({"celltype": "T",
                        "Sample": rng.permutation(["ctrl"] * 400 + ["stim"] * 400)},
                       index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X=sp.csr_matrix(X), obs=obs,
                   var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))
    d = _run(a, tmp_path)
    t = d[d.tested == True]                                         # noqa: E712
    assert (t.fdr < 0.05).sum() <= 0.02 * len(t), "type-I error too high on null data"


def test_back_compat_aliases_exist():
    """The scPyDE -> DUET rename promised these keep working."""
    import duet
    for name in ("run_duet", "run_duet_cnv", "run_scpyde", "run_python_hurdle_de"):
        assert hasattr(duet, name), f"{name} disappeared from the public API"


# --------------------------------------------------------------------------
# quasi-separation: the coefficient has no MLE, but the LRT is still valid
# --------------------------------------------------------------------------

def test_is_separated_is_exact_not_heuristic():
    from duet.core import _is_separated
    cond = np.r_[np.zeros(50), np.ones(50)]
    assert _is_separated(np.r_[np.ones(50), np.zeros(50)], cond)      # 100% / 0%
    assert _is_separated(np.r_[np.ones(50), np.r_[np.ones(25),
                                                  np.zeros(25)]], cond)  # 100% / 50%
    assert not _is_separated(np.r_[np.ones(30), np.zeros(20),
                                   np.ones(10), np.zeros(40)], cond)     # 60% / 20%


def _separated_toy(tmp_path, seed=3):
    """One gene detected in 100% of the reference arm -- quasi-separation."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    n, g = 400, 30
    half = n // 2
    X = (rng.random((n, g)) < 0.5) * rng.lognormal(1.5, 0.5, (n, g))
    X[:half, 0] = rng.lognormal(1.5, 0.5, half)      # ref: every cell detects
    X[half:, 0] = (rng.random(half) < 0.6) * rng.lognormal(1.5, 0.5, half)
    obs = pd.DataFrame({"celltype": "T",
                        "Sample": ["ctrl"] * half + ["stim"] * half},
                       index=[f"c{i}" for i in range(n)])
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs,
                      var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))


def test_separated_gene_is_flagged(tmp_path):
    a = _separated_toy(tmp_path)
    d = _run(a, tmp_path, covariates=("CDR",), mast_compat=False).set_index("gene")
    assert bool(d.loc["g0", "detection_separated"]) is True
    assert d.loc["g0", "detect_rate_ref"] == 1.0
    # the rest of the genes have interior detection rates and must not be flagged
    others = d.drop(index="g0")
    assert not others["detection_separated"].fillna(False).any()


def test_separation_does_not_invalidate_the_test(tmp_path):
    """The LRT and p-value stay finite and usable under separation; only the
    coefficient is undefined. A NaN statistic here would be a regression."""
    a = _separated_toy(tmp_path)
    d = _run(a, tmp_path, covariates=("CDR",), mast_compat=False).set_index("gene")
    r = d.loc["g0"]
    assert bool(r["tested"]) is True
    assert np.isfinite(r["stat_hurdle"]) and r["stat_hurdle"] > 0
    assert np.isfinite(r["neglog10p"])


def test_engines_agree_on_separated_genes(tmp_path):
    """Regression test for a real bug: separation used to be diagnosed from
    |beta| > 25, so the two engines took DIFFERENT tests on the same gene
    depending on where their iteration stopped (|beta| 24.6 vs 27.1)."""
    a = _separated_toy(tmp_path)
    sm = _run(a, tmp_path / "sm", covariates=("CDR",), mast_compat=False,
              logistic_engine="statsmodels").set_index("gene")
    ni = _run(a, tmp_path / "ni", covariates=("CDR",), mast_compat=False,
              logistic_engine="numpy_irls").set_index("gene")
    m = (sm.tested == True) & (ni.reindex(sm.index).tested == True)   # noqa: E712
    assert np.allclose(sm.loc[m, "stat_hurdle"],
                       ni.reindex(sm.index).loc[m, "stat_hurdle"],
                       rtol=1e-6, atol=1e-6)


def test_neglog10p_is_consistent_with_its_own_pvalue(toy, tmp_path):
    """Regression: empirical-Bayes moderation rewrote stat_hurdle and pvalue but
    left neglog10p holding the PRE-moderation value, so the column the docs tell
    users to rank on disagreed with the p-value beside it by up to 0.43."""
    from scipy.stats import chi2
    for kw in ({"mast_compat": True},                       # vectorised path
               {"covariates": ("CDR",), "eb_shrinkage": True}):   # general path
        d = _run(toy, tmp_path / str(abs(hash(str(kw)))), **kw)
        t = d[(d.tested == True) & (d.pvalue > 0)]          # noqa: E712
        assert len(t) > 10
        implied = -chi2.logsf(t.stat_hurdle.to_numpy(float),
                              t.df_hurdle.to_numpy(float)) / np.log(10)
        assert np.allclose(implied, t.neglog10p.to_numpy(float), atol=1e-9), \
            f"neglog10p inconsistent with stat_hurdle for {kw}"


# --------------------------------------------------------------------------
# pseudobulk arm: the aggregation is what has to be right
# --------------------------------------------------------------------------

def _donor_toy(n_donors=6, per=60, n_genes=40, paired=False, seed=11):
    """Counts with a donor structure, so aggregation can be checked exactly."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    donors, conds = [], []
    for d in range(n_donors):
        if paired:                       # every donor gives both arms
            donors += [f"d{d}"] * per
            conds += ["ctrl"] * (per // 2) + ["stim"] * (per - per // 2)
        else:                            # each donor belongs to one arm
            donors += [f"d{d}"] * per
            conds += [("ctrl" if d < n_donors // 2 else "stim")] * per
    n = len(donors)
    X = rng.poisson(3.0, size=(n, n_genes)).astype(float)
    stim = np.array(conds) == "stim"
    X[stim, :5] *= 6                     # genes 0-4 are genuinely differential
    obs = pd.DataFrame({"celltype": "T", "donor": donors, "condition": conds},
                       index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X=sp.csr_matrix(np.log1p(X)), obs=obs,
                   var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)]))
    a.layers["counts"] = sp.csr_matrix(X)
    return a


def test_pseudobulk_aggregates_on_donor_by_condition():
    """The unit is (donor, condition). Grouping by donor alone would merge a
    paired design into one profile per donor and lose the contrast entirely."""
    from duet import aggregate_pseudobulk
    a = _donor_toy(n_donors=4, paired=True)
    pb, meta = aggregate_pseudobulk(
        a, donor_col="donor", condition_col="condition",
        ref_label="ctrl", test_label="stim")
    assert len(pb) == 8, "4 donors x 2 conditions must give 8 profiles"
    assert (meta["condition"] == "ctrl").sum() == 4
    assert (meta["condition"] == "stim").sum() == 4
    assert meta.index.is_unique


def test_pseudobulk_counts_are_exact_sums():
    from duet import aggregate_pseudobulk
    import scipy.sparse as sp
    a = _donor_toy(n_donors=4)
    pb, meta = aggregate_pseudobulk(
        a, donor_col="donor", condition_col="condition",
        ref_label="ctrl", test_label="stim")
    raw = a.layers["counts"]
    raw = raw.toarray() if sp.issparse(raw) else np.asarray(raw)
    for name, row in meta.iterrows():
        m = ((a.obs.donor == row.donor) & (a.obs.condition == row.condition)).to_numpy()
        assert np.allclose(pb.loc[name].to_numpy(), raw[m].sum(axis=0))


def test_pseudobulk_needs_a_counts_layer():
    """X is log-normalised; a negative-binomial model on it would be nonsense."""
    from duet import aggregate_pseudobulk
    a = _donor_toy()
    del a.layers["counts"]
    with pytest.raises(ValueError, match="raw counts"):
        aggregate_pseudobulk(a, donor_col="donor", condition_col="condition",
                             ref_label="ctrl", test_label="stim")


def test_pseudobulk_refuses_too_few_donors():
    """Too few samples must be reported, not returned as a table of NaN that
    reads as 'nothing was differential'."""
    from duet import run_pseudobulk
    pytest.importorskip("pydeseq2")
    a = _donor_toy(n_donors=2)           # one donor per arm
    r = run_pseudobulk(a, donor_col="donor", condition_col="condition",
                       ref_label="ctrl", test_label="stim",
                       min_donors_per_group=2)
    assert not r["pb_tested"].any()
    assert r["pb_skip_reason"].str.contains("donors/group").all()


def test_combine_calls_requires_the_sample_level_arm():
    """A gene the hurdle calls but pseudobulk does not is ranked_only, never
    significant -- that is the whole point of the second arm."""
    import pandas as pd
    from duet import combine_calls
    duet_df = pd.DataFrame({"gene": ["a", "b", "c"], "celltype": "T",
                            "fdr": [0.001, 0.001, 0.9]})
    pb_df = pd.DataFrame({"gene": ["a", "b", "c"], "celltype": "T",
                          "pb_padj": [0.01, 0.5, 0.5], "pb_tested": True})
    out = combine_calls(duet_df, pb_df).set_index("gene")
    assert out.loc["a", "call"] == "significant"
    assert out.loc["b", "call"] == "ranked_only"
    assert out.loc["c", "call"] == "ns"


def test_donor_and_condition_must_differ():
    from duet import run_duet_pseudobulk
    a = _donor_toy()
    with pytest.raises(ValueError, match="must differ"):
        run_duet_pseudobulk(a, condition_col="condition", donor_col="condition",
                            ref_label="ctrl", test_label="stim")
