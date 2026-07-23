"""
run_de_comparison.py
====================
Run several Python single-cell differential-expression methods on the SAME data
(Reference vs pancreas, per cell type), save one harmonized result file per
method, then cross-compare the methods gene-by-gene.

Methods: DUET (the two-part hurdle in ./duet), diffxpy (Wald), DESeq2 (pydeseq2 pseudobulk),
scanpy Wilcoxon.

Each method's output is harmonized to the SAME columns:
    method, celltype, comparison, gene, pvalue, fdr, log2fc, neglog10p, stat,
    n_ref, n_test, tested, skip_reason
with positive log2fc = UP in pancreas (test) relative to Reference.

Run:  python run_de_comparison.py                       # default 3 cell types, all genes
      python run_de_comparison.py --max-genes 2000 --n-celltypes 1   # fast smoke test
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `from duet import ...` works

COMPARISON = "Reference_vs_pancreas"
HARMON = ["method", "celltype", "comparison", "gene", "pvalue", "fdr", "log2fc",
          "neglog10p", "stat", "n_ref", "n_test", "tested", "skip_reason"]
LN2 = math.log(2.0)
LN10 = math.log(10.0)
# -log10 of the smallest positive float64 (~5e-324): the hard floor a p-value can
# reach before it is stored as exactly 0.0.
NLP_FLOOR = 323.0


def _nlp_from_normal(pvalue, z):
    """Underflow-safe -log10(p) for a two-sided normal/Wald test.

    Used for Wilcoxon (scanpy `scores`) and DESeq2 (Wald z). Where the stored
    p-value underflowed to 0.0 we rebuild the tail from the z statistic via
    ``norm.logsf``, which stays finite far below the float64 p-value floor.
    """
    from scipy.stats import norm
    p = np.asarray(pvalue, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p) & (p > 0)
    out[ok] = -np.log10(p[ok])
    dead = np.isfinite(p) & (p <= 0)
    z = np.asarray(z, dtype=float)
    use = dead & np.isfinite(z)
    if use.any():
        out[use] = -(math.log(2.0) + norm.logsf(np.abs(z[use]))) / LN10
    out[dead & ~use] = NLP_FLOOR
    return out


def _nlp_naive(pvalue):
    """-log10(p) with no recovery possible: the method's harmonized `stat` is not a
    test statistic (diffxpy stores `coef_mle`) or is absent entirely (the R MAST
    CSV carries none), so an underflowed p-value is floored at 323 and its rank
    within the tail is genuinely unrecoverable from this output."""
    p = np.asarray(pvalue, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p) & (p > 0)
    out[ok] = -np.log10(p[ok])
    out[np.isfinite(p) & (p <= 0)] = NLP_FLOOR
    return out


_LOG_FH = None  # optional file handle (set in main) -> reliable progress log


def _log(msg: str) -> None:
    line = f"[de-compare][{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(line + "\n"); _LOG_FH.flush()
        except Exception:
            pass


def _empty(method: str, celltype: str, n_ref: int, n_test: int, reason: str) -> pd.DataFrame:
    return pd.DataFrame([{**{c: np.nan for c in HARMON}, "method": method, "celltype": celltype,
                          "comparison": COMPARISON, "n_ref": n_ref, "n_test": n_test,
                          "tested": False, "skip_reason": reason}])


# --------------------------------------------------------------------------- #
# Per-method runners. Each returns a harmonized DataFrame (HARMON columns).
# --------------------------------------------------------------------------- #
def run_duet(ad_ct, celltype, *, sample_col, ref_label, test_label, n_ref, n_test,
             min_detect_frac=0.10, linear_logfc=False):
    from duet import run_duet as _run_duet
    with tempfile.TemporaryDirectory() as tmp:
        out = _run_duet(
            ad_ct, output_dir=tmp, celltype_col="celltype", sample_col=sample_col,
            ref_label=ref_label, test_label=test_label, mast_compat=True,
            eb_shrinkage=True, vectorized=True, logistic_engine="numpy_irls",
            memory_log=False, output_name="duet.csv",
            # DEFAULT compressed log-mean-diff coef (resembles R MAST); flip to the
            # scanpy/Wilcoxon-style linear log2FC with linear_logfc=True.
            linear_logfc=bool(linear_logfc),
        )
        df = pd.read_csv(out)
    df = df[df["tested"] == True].copy()  # noqa: E712
    # min.pct filter (Seurat/scanpy standard): the hurdle's detection component
    # flags genes detected in only a tiny fraction of cells (a 18%->21% detection
    # shift is "significant" at thousands of cells), which is why DUET reports
    # ~3x more genes than the others. Keep only genes detected in >= min_detect_frac
    # of cells in AT LEAST ONE group -> removes those lowly-detected calls.
    if min_detect_frac and min_detect_frac > 0:
        maxdet = df[["detect_rate_ref", "detect_rate_test"]].max(axis=1)
        df = df[maxdet >= float(min_detect_frac)].copy()
    # linear coef is already log2; compressed coef is a natural-log mean diff -> /ln2
    log2fc = df["coef"].astype(float) if linear_logfc else df["coef"].astype(float) / LN2
    res = pd.DataFrame({
        "method": "DUET", "celltype": celltype, "comparison": COMPARISON,
        "gene": df["gene"].astype(str),
        "pvalue": df["pvalue"].astype(float),
        "fdr": df["fdr"].astype(float),
        "log2fc": log2fc,
        # DUET is the one method here that can report an exact tail: the hurdle
        # statistic is chi2 with a known df, so -log10 p is recoverable no matter
        # how far below the float64 floor the true p-value sits.
        "neglog10p": df["neglog10p"].astype(float),
        "stat": df["stat_hurdle"].astype(float),
        "n_ref": n_ref, "n_test": n_test, "tested": True, "skip_reason": "",
    })
    return res[HARMON]


_DASK_PATCHED = False


def _patch_dask_auto_chunks():
    """diffxpy/batchglm wrap the data as dask arrays; in modern dask (>=2025) the
    `auto_chunks` helper computes `... ** (1/len(autos))` and raises
    ZeroDivisionError when NO dimension is 'auto' -- which batchglm triggers in its
    Fisher-information-matrix concatenate once there are more than a few hundred
    genes (works at 150 genes, crashes at 2000). We guard it: no 'auto' dims -> the
    chunks are already concrete, and on any residual ZeroDivisionError fall back to
    one full-size chunk per auto dim. Idempotent; affects only the diffxpy path."""
    global _DASK_PATCHED
    if _DASK_PATCHED:
        return
    try:
        import dask.array.core as _dac
        _orig = _dac.auto_chunks

        def _safe(chunks, shape, limit, dtype, previous_chunks=None):
            if not any(c == "auto" for c in chunks):
                return tuple(chunks)
            try:
                return _orig(chunks, shape, limit, dtype, previous_chunks)
            except ZeroDivisionError:
                return tuple((shape[i],) if c == "auto" else c for i, c in enumerate(chunks))

        _dac.auto_chunks = _safe
        _DASK_PATCHED = True
    except Exception:
        pass


def run_diffxpy(ad_ct, celltype, *, counts_layer, condition, ref_label, test_label,
                n_ref, n_test, min_cells, max_cells_per_group=5000, max_genes=8000,
                nproc=1, seed=0):
    import anndata as ad
    _patch_dask_auto_chunks()
    import diffxpy.api as de

    # RAM-safety: diffxpy 0.7.4 needs a DENSE float64 matrix (it mishandles sparse:
    # converts to pydata-sparse COO -> numpy.take TypeError). Densifying a whole cell
    # type at full gene count is huge (60k cells x 59k genes x 8B ~ 28 GB), so we
    # (1) subsample cells per group and (2) filter genes -- BOTH on the SPARSE matrix
    # -- and densify ONLY the resulting bounded submatrix.
    counts = ad_ct.layers[counts_layer]
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    cond = np.asarray(condition).astype(str)

    rng = np.random.default_rng(seed)
    rows = []
    for lab in (ref_label, test_label):
        idx = np.where(cond == lab)[0]
        if max_cells_per_group and idx.size > max_cells_per_group:
            idx = rng.choice(idx, max_cells_per_group, replace=False)
        rows.append(idx)
    rows = np.sort(np.concatenate(rows))
    Xs = counts[rows]
    cond_s = cond[rows]

    # gene filter on sparse: expressed in >= min_cells of the (subsampled) cells
    nnz_per_gene = np.asarray((Xs > 0).sum(axis=0)).ravel()
    gkeep = np.where(nnz_per_gene >= min_cells)[0]
    if gkeep.size == 0:
        return _empty("diffxpy", celltype, n_ref, n_test, "no_genes_after_filter")
    # cap to the top-N expressed genes: bounds runtime AND RAM (batchglm copies the
    # dense data per worker). The other methods still test all genes.
    if max_genes and gkeep.size > max_genes:
        tot = np.asarray(Xs[:, gkeep].sum(axis=0)).ravel()
        gkeep = gkeep[np.sort(np.argsort(tot)[::-1][:max_genes])]
    Xs = Xs[:, gkeep]
    genes = ad_ct.var_names.to_numpy().astype(str)[gkeep]

    # diffxpy's size factors divide by per-cell total counts -> drop zero-count cells
    ckeep = np.asarray(Xs.sum(axis=1)).ravel() > 0
    Xs = Xs[ckeep]; cond_s = cond_s[ckeep]
    n_ref_used = int((cond_s == ref_label).sum()); n_test_used = int((cond_s == test_label).sum())
    if n_ref_used < min_cells or n_test_used < min_cells:
        return _empty("diffxpy", celltype, n_ref, n_test, "too_few_cells_after_filter")

    # densify ONLY the bounded submatrix (<=2*max_cells_per_group rows x kept genes)
    sub = ad.AnnData(
        X=np.ascontiguousarray(Xs.toarray(), dtype=np.float64),
        obs=pd.DataFrame({"condition": pd.Categorical(cond_s, categories=[ref_label, test_label])}),
        var=pd.DataFrame(index=genes))
    # nproc=1: batchglm's dispersion fitting otherwise spawns `nproc` workers that
    # EACH copy the dense data (Windows spawn) -> RAM blow-up. Single process is
    # slower but predictable and RAM-bounded.
    test = de.test.wald(data=sub, formula_loc="~1+condition", factor_loc_totest="condition",
                        train_args={"nproc": int(nproc)})
    s = test.summary()
    res = pd.DataFrame({
        "method": "diffxpy", "celltype": celltype, "comparison": COMPARISON,
        "gene": s["gene"].astype(str),
        "pvalue": s["pval"].astype(float),
        "fdr": s["qval"].astype(float),
        "log2fc": s["log2fc"].astype(float),
        # NOT recoverable: `stat` below holds coef_mle, a COEFFICIENT, not a test
        # statistic -- and 41-54% of diffxpy's p-values underflow to 0.0, so its
        # tail ordering is lost. Rank metrics against diffxpy are unreliable.
        "neglog10p": _nlp_naive(s["pval"].astype(float)),
        "stat": s.get("coef_mle", pd.Series(np.nan, index=s.index)).astype(float),
        "n_ref": n_ref_used, "n_test": n_test_used, "tested": True, "skip_reason": "",
    })
    return res[HARMON]


def run_deseq2(ad_ct, celltype, *, counts_layer, sample_col, condition, ref_label,
               test_label, n_ref, n_test, min_samples_per_group, min_gene_counts,
               min_cells_per_sample):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    counts = ad_ct.layers[counts_layer]
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    samples = ad_ct.obs[sample_col].astype(str).to_numpy()
    cond_by_cell = np.asarray(condition)

    # ---- pseudobulk: sum raw counts per (SourceFile, condition) ----
    # The unit MUST be sample x condition, not sample alone. In a PAIRED design
    # (e.g. Kang 2018: every donor contributes both ctrl and stim cells) grouping
    # by sample alone merges the two conditions into one pseudobulk sample and
    # then takes the condition from an arbitrary cell, which silently collapses
    # every sample into one group. Datasets where each sample belongs entirely to
    # one condition are unaffected: the grouping is then identical.
    rows, smeta = [], []
    genes = ad_ct.var_names.astype(str).to_numpy()
    for s in pd.unique(samples):
        for lab in (ref_label, test_label):
            m = (samples == s) & (cond_by_cell == lab)
            if int(m.sum()) < min_cells_per_sample:
                continue
            rows.append(np.asarray(counts[m].sum(axis=0)).ravel())
            # name stays unique when a sample appears in both conditions
            smeta.append((s if (samples == s).sum() == m.sum() else f"{s}|{lab}", lab))
    if len(rows) < 2:
        return _empty("DESeq2", celltype, n_ref, n_test, "too_few_pseudobulk_samples")
    pb = pd.DataFrame(np.rint(np.vstack(rows)).astype(int),
                      index=[s for s, _ in smeta], columns=genes)
    meta = pd.DataFrame({"condition": pd.Categorical([c for _, c in smeta],
                                                     categories=[ref_label, test_label])},
                        index=pb.index)
    n_ref_s = int((meta["condition"] == ref_label).sum())
    n_test_s = int((meta["condition"] == test_label).sum())
    if n_ref_s < min_samples_per_group or n_test_s < min_samples_per_group:
        return _empty("DESeq2", celltype, n_ref, n_test,
                      f"need>={min_samples_per_group}/group (ref={n_ref_s},test={n_test_s})")

    pb = pb.loc[:, pb.sum(axis=0) >= min_gene_counts]  # drop low-count genes
    if pb.shape[1] == 0:
        return _empty("DESeq2", celltype, n_ref, n_test, "no_genes_after_count_filter")

    dds = DeseqDataSet(counts=pb, metadata=meta, design="~condition", quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", test_label, ref_label], quiet=True)
    st.summary()
    r = st.results_df.reset_index().rename(columns={"index": "gene"})
    gcol = "gene" if "gene" in r.columns else r.columns[0]
    res = pd.DataFrame({
        "method": "DESeq2", "celltype": celltype, "comparison": COMPARISON,
        "gene": r[gcol].astype(str),
        "pvalue": r["pvalue"].astype(float),
        "fdr": r["padj"].astype(float),
        "log2fc": r["log2FoldChange"].astype(float),
        "neglog10p": _nlp_from_normal(r["pvalue"].astype(float), r["stat"].astype(float)),
        "stat": r["stat"].astype(float),
        "n_ref": n_ref, "n_test": n_test, "tested": True, "skip_reason": "",
    })
    return res[HARMON]


def run_wilcoxon(ad_ct, celltype, *, ref_label, test_label, n_ref, n_test):
    import scanpy as sc
    import anndata as ad
    # build a minimal AnnData that REFERENCES ad_ct.X (no deep copy of X + the counts
    # layer) -- rank_genes_groups only reads X and writes results to .uns
    a = ad.AnnData(
        X=ad_ct.X,
        obs=pd.DataFrame({"condition": pd.Categorical(ad_ct.obs["condition"].astype(str),
                                                      categories=[ref_label, test_label])},
                         index=ad_ct.obs_names),
        var=pd.DataFrame(index=ad_ct.var_names))
    sc.tl.rank_genes_groups(a, "condition", groups=[test_label], reference=ref_label,
                            method="wilcoxon")
    d = sc.get.rank_genes_groups_df(a, group=test_label)
    res = pd.DataFrame({
        "method": "wilcoxon", "celltype": celltype, "comparison": COMPARISON,
        "gene": d["names"].astype(str),
        "pvalue": d["pvals"].astype(float),
        "fdr": d["pvals_adj"].astype(float),
        "log2fc": d["logfoldchanges"].astype(float),
        # scanpy's `scores` is a two-sided normal z -> the tail IS recoverable.
        "neglog10p": _nlp_from_normal(d["pvals"].astype(float), d["scores"].astype(float)),
        "stat": d["scores"].astype(float),
        "n_ref": n_ref, "n_test": n_test, "tested": True, "skip_reason": "",
    })
    return res[HARMON]


METHODS = {
    "duet": run_duet,
    "scpyde": run_duet,          # deprecated alias for --methods
    "diffxpy": run_diffxpy,
    "deseq2": run_deseq2,
    "wilcoxon": run_wilcoxon,
}


# --------------------------------------------------------------------------- #
# Cell-type selection.
# --------------------------------------------------------------------------- #
def pick_celltypes(obs, *, celltype_col, condition, ref_label, test_label, sample_col,
                   n_celltypes, min_cells, min_samples_per_group):
    rows = []
    cond = pd.Series(condition, index=obs.index)
    for ct in obs[celltype_col].astype(str).unique():
        m = obs[celltype_col].astype(str) == ct
        n_ref = int((cond[m] == ref_label).sum())
        n_test = int((cond[m] == test_label).sum())
        sref = obs.loc[m & (cond == ref_label), sample_col].nunique()
        stest = obs.loc[m & (cond == test_label), sample_col].nunique()
        eligible = (n_ref >= min_cells and n_test >= min_cells
                    and sref >= min_samples_per_group and stest >= min_samples_per_group)
        rows.append((ct, n_ref, n_test, sref, stest, eligible, min(n_ref, n_test)))
    df = pd.DataFrame(rows, columns=["celltype", "n_ref", "n_test", "s_ref", "s_test",
                                     "eligible", "balance"])
    elig = df[df["eligible"]].sort_values("balance", ascending=False)
    return elig.head(n_celltypes)["celltype"].tolist(), df.sort_values("balance", ascending=False)


# --------------------------------------------------------------------------- #
# Comparison / concordance.
# --------------------------------------------------------------------------- #
def _corr(x, y, kind):
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    try:
        f = pearsonr if kind == "p" else spearmanr
        return round(float(f(x[ok], y[ok])[0]), 3)
    except Exception:
        return np.nan


def build_comparison(all_df, *, fdr_thr, top_k, out_dir):
    methods = sorted(all_df["method"].unique())
    # ---- per-gene wide table ----
    wide = None
    for m in methods:
        cols = ["celltype", "gene", "pvalue", "fdr", "log2fc"]
        if "neglog10p" in all_df.columns:
            cols.append("neglog10p")
        sub = all_df[(all_df.method == m) & (all_df.tested == True)][cols].rename(  # noqa: E712
            columns={"pvalue": f"pvalue_{m}", "fdr": f"fdr_{m}", "log2fc": f"log2fc_{m}",
                     "neglog10p": f"neglog10p_{m}"})
        wide = sub if wide is None else wide.merge(sub, on=["celltype", "gene"], how="outer")
    if wide is None or wide.empty:
        # no method produced tested genes -> write empty (but headed) outputs, don't crash
        cols = ["celltype", "gene"] + [f"{p}_{m}" for m in methods for p in ("pvalue", "fdr", "log2fc")]
        pd.DataFrame(columns=cols).to_csv(out_dir / "comparison_per_gene.csv", index=False)
        pd.DataFrame(columns=["celltype", "method_a", "method_b"]).to_csv(out_dir / "comparison_concordance.csv", index=False)
        pd.DataFrame(columns=["celltype", "gene", "n_methods_sig", "methods_sig", "same_direction"]).to_csv(out_dir / "consensus_genes.csv", index=False)
        (out_dir / "comparison_summary.txt").write_text(
            "DE method comparison\n====================\n\n"
            f"methods attempted: {methods}\n\nNo tested genes were available to compare "
            "(every method skipped or failed). See the per-method CSVs / logs.\n", encoding="utf-8")
        print("[de-compare] WARNING: no tested genes across methods -> empty comparison outputs", flush=True)
        return
    wide.to_csv(out_dir / "comparison_per_gene.csv", index=False)

    # ---- pairwise concordance per cell type + pooled ----
    rows = []
    for ct in sorted(all_df["celltype"].dropna().unique()):
        scope = [(ct, wide[wide.celltype == ct])]
        if ct == sorted(all_df["celltype"].dropna().unique())[0]:
            scope = scope  # (pooled added once below)
        for i, ma in enumerate(methods):
            for mb in methods[i + 1:]:
                w = wide[wide.celltype == ct]
                both = w.dropna(subset=[f"log2fc_{ma}", f"log2fc_{mb}"])
                if len(both) < 5:
                    continue
                sig_a = set(both.loc[both[f"fdr_{ma}"] < fdr_thr, "gene"])
                sig_b = set(both.loc[both[f"fdr_{mb}"] < fdr_thr, "gene"])
                jac = len(sig_a & sig_b) / len(sig_a | sig_b) if (sig_a | sig_b) else np.nan
                # Rank on neglog10p, NOT on pvalue. At these cell counts a few
                # percent of p-values underflow to exactly 0.0; ranking on the raw
                # p-value then breaks those ties arbitrarily (by row order), which
                # shifts top-K overlaps by up to 46 genes out of 100. neglog10p is
                # exact for scPyDE/DUET and Wilcoxon, floored for the rest.
                _nlp_a = f"neglog10p_{ma}" if f"neglog10p_{ma}" in both.columns else None
                _nlp_b = f"neglog10p_{mb}" if f"neglog10p_{mb}" in both.columns else None
                if _nlp_a and _nlp_b:
                    tA = set(both.nlargest(top_k, _nlp_a).gene)
                    tB = set(both.nlargest(top_k, _nlp_b).gene)
                    rank_a, rank_b = both[_nlp_a], both[_nlp_b]
                else:
                    tA = set(both.nsmallest(top_k, f"pvalue_{ma}").gene)
                    tB = set(both.nsmallest(top_k, f"pvalue_{mb}").gene)
                    rank_a = -np.log10(both[f"pvalue_{ma}"].clip(1e-300))
                    rank_b = -np.log10(both[f"pvalue_{mb}"].clip(1e-300))
                rows.append({
                    "celltype": ct, "method_a": ma, "method_b": mb, "n_common": len(both),
                    "log2fc_pearson": _corr(both[f"log2fc_{ma}"], both[f"log2fc_{mb}"], "p"),
                    "log2fc_spearman": _corr(both[f"log2fc_{ma}"], both[f"log2fc_{mb}"], "s"),
                    "sign_agree_pct": round(100 * float((np.sign(both[f"log2fc_{ma}"]) ==
                                                         np.sign(both[f"log2fc_{mb}"])).mean()), 1),
                    "neglog10p_spearman": _corr(rank_a, rank_b, "s"),
                    f"top{top_k}_overlap": len(tA & tB),
                    "sig_a": len(sig_a), "sig_b": len(sig_b), "sig_jaccard": round(jac, 3) if jac == jac else np.nan,
                })
    conc = pd.DataFrame(rows)
    conc.to_csv(out_dir / "comparison_concordance.csv", index=False)

    # ---- consensus per gene (significant in how many methods, same direction) ----
    cons_rows = []
    for _, row in wide.iterrows():
        sig = [m for m in methods if pd.notna(row.get(f"fdr_{m}")) and row[f"fdr_{m}"] < fdr_thr]
        signs = [np.sign(row[f"log2fc_{m}"]) for m in sig if pd.notna(row.get(f"log2fc_{m}"))]
        same_dir = len(set(signs)) <= 1 and len(signs) > 0
        cons_rows.append({"celltype": row["celltype"], "gene": row["gene"],
                          "n_methods_sig": len(sig), "methods_sig": ",".join(sig),
                          "same_direction": bool(same_dir)})
    cons = pd.DataFrame(cons_rows)
    cons.sort_values(["n_methods_sig"], ascending=False).to_csv(out_dir / "consensus_genes.csv", index=False)

    # ---- summary text ----
    lines = ["DE method comparison — concordance summary", "=" * 50, ""]
    lines.append(f"methods: {methods}")
    lines.append(f"cell types: {sorted(all_df['celltype'].dropna().unique())}")
    lines.append("")
    lines.append("Significant genes (fdr<%.2f) per method per cell type:" % fdr_thr)
    sigtab = (all_df[all_df.tested == True].assign(sig=lambda d: d.fdr < fdr_thr)  # noqa: E712
              .groupby(["celltype", "method"])["sig"].sum().unstack(fill_value=0))
    lines.append(sigtab.to_string())
    lines.append("")
    lines.append("Pairwise concordance (per cell type):")
    if not conc.empty:
        lines.append(conc.to_string(index=False))
    lines.append("")
    n_all = int((cons["n_methods_sig"] == len(methods)).sum())
    lines.append(f"Consensus DE genes (significant in ALL {len(methods)} methods): {n_all}")
    lines.append("")
    lines.append("CAVEATS:")
    lines.append("  * DESeq2 = pseudobulk (sample-level, proper replication) -> far fewer 'significant'")
    lines.append("    genes than the per-cell methods (huge N -> tiny p). Judge by rank/direction, not counts.")
    lines.append("  * log2fc bases differ; DUET's default is the COMPRESSED MAST-style log-mean-diff (/ln2).")
    lines.append("  * Significant-gene OVERLAP (Jaccard) is indicative; the robust metrics are")
    lines.append("    log2fc sign agreement, log2fc Spearman, and p-value rank (neglog10p) Spearman.")
    (out_dir / "comparison_summary.txt").write_text("\n".join(map(str, lines)), encoding="utf-8")
    return conc


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare single-cell DE methods on the same data.")
    ap.add_argument("--h5ad", default=str(HERE / "PancreasIntegratedAnnotated.h5ad"))
    ap.add_argument("--output-dir", default=str(HERE / "results"))
    ap.add_argument("--celltypes", default=None, help="Comma-separated; default = auto-pick balanced.")
    ap.add_argument("--n-celltypes", type=int, default=3)
    ap.add_argument("--celltype-col", default="celltype")
    ap.add_argument("--sample-col", default="Sample")
    ap.add_argument("--source-col", default="SourceFile", help="Replicate column for DESeq2 pseudobulk.")
    ap.add_argument("--ref-label", default="Reference")
    ap.add_argument("--test-label", default="pancreas")
    ap.add_argument("--counts-layer", default="counts")
    ap.add_argument("--methods", default="duet,diffxpy,deseq2,wilcoxon",
                    help="Comma-separated. 'scpyde' is accepted as a deprecated alias for 'duet'.")
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--duet-min-detect-frac", "--scpyde-min-detect-frac", type=float, default=0.10,
                    dest="duet_min_detect_frac",
                    help="DUET min.pct: keep only genes detected in >= this fraction of cells "
                         "in >=1 group (Seurat/scanpy standard; 0 = off -> the full hurdle output).")
    ap.add_argument("--duet-linear-logfc", "--scpyde-linear-logfc", action="store_true",
                    dest="duet_linear_logfc",
                    help="DUET reports the LINEAR (scanpy/Wilcoxon) log2FC. Default OFF -> "
                         "COMPRESSED MAST-style log-mean-diff (resembles R MAST).")
    ap.add_argument("--min-samples-per-group", type=int, default=2)
    ap.add_argument("--min-cells-per-sample", type=int, default=10)
    ap.add_argument("--min-gene-counts", type=int, default=10)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--max-genes", type=int, default=None, help="Cap genes (smoke test).")
    ap.add_argument("--diffxpy-max-cells-per-group", type=int, default=5000,
                    help="diffxpy needs a DENSE matrix; subsample to this many cells/group "
                         "to bound RAM (0 = use all cells). Other methods always use all cells.")
    ap.add_argument("--diffxpy-max-genes", type=int, default=8000,
                    help="diffxpy tests only the top-N expressed genes (bounds runtime+RAM; "
                         "0 = all genes). Other methods test all genes.")
    ap.add_argument("--diffxpy-nproc", type=int, default=1,
                    help="diffxpy/batchglm worker processes (1 = lowest RAM, no per-worker data copy).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (diffxpy cell subsample).")
    args = ap.parse_args(argv)

    import anndata as ad
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    global _LOG_FH
    try:
        _LOG_FH = open(out_dir / "run.log", "a", encoding="utf-8")
    except Exception:
        _LOG_FH = None

    _log(f"loading (backed): {args.h5ad}")
    A = ad.read_h5ad(args.h5ad, backed="r")
    obs = A.obs
    samp = obs[args.sample_col].astype(str)
    condition_all = np.where(samp.str.contains(args.ref_label, case=False, na=False),
                             args.ref_label, args.test_label)
    n_ref_all = int((condition_all == args.ref_label).sum())
    if n_ref_all == 0 or n_ref_all == len(condition_all):
        raise SystemExit(
            f"[de-compare] --ref-label '{args.ref_label}' matched {n_ref_all} of "
            f"{len(condition_all)} cells in obs['{args.sample_col}'] "
            f"(values: {sorted(set(samp))[:8]}). Condition is derived by a "
            f"case-insensitive SUBSTRING match, so the ref label must appear in the "
            f"reference group's value and in no other.")

    # The comparison tag follows the actual labels, so results from different
    # datasets (ctrl_vs_stim, Vehicle_vs_LPS, ...) are not all stamped with the
    # pancreas one. Rebinds the module global the per-method runners read.
    global COMPARISON
    COMPARISON = f"{args.ref_label}_vs_{args.test_label}"
    _log(f"comparison: {COMPARISON}  (ref={n_ref_all} cells, "
         f"test={len(condition_all) - n_ref_all} cells)")

    if args.celltypes:
        celltypes = [c.strip() for c in args.celltypes.split(",") if c.strip()]
    else:
        celltypes, ranking = pick_celltypes(
            obs, celltype_col=args.celltype_col, condition=condition_all,
            ref_label=args.ref_label, test_label=args.test_label, sample_col=args.source_col,
            n_celltypes=args.n_celltypes, min_cells=args.min_cells,
            min_samples_per_group=args.min_samples_per_group)
        _log(f"auto-picked cell types: {celltypes}")
    _log(f"methods: {methods}")

    all_results = []
    ct_labels = obs[args.celltype_col].astype(str).to_numpy()
    for ct in celltypes:
        mask = ct_labels == ct
        _log(f"=== cell type '{ct}': {int(mask.sum())} cells -> materializing subset ===")
        # materialize ONLY X + counts for this cell type (the DE methods need nothing
        # else). Skips the heavy obsm (X_cnv/X_scVI/ora_*) + obsp (cell x cell graphs)
        # that A[mask].to_memory() would otherwise pull into RAM.
        view = A[mask]
        ad_ct = ad.AnnData(X=view.X, obs=view.obs.copy(), var=A.var.copy())
        if args.counts_layer in A.layers:
            ad_ct.layers[args.counts_layer] = view.layers[args.counts_layer]
        ad_ct.obs["condition"] = condition_all[mask]
        if args.max_genes and ad_ct.n_vars > args.max_genes:
            # smoke-test cap: keep the TOP-expressed genes (representative; the first
            # N genes alphabetically are mostly all-zero -> breaks diffxpy size factors)
            src = ad_ct.layers[args.counts_layer] if args.counts_layer in ad_ct.layers else ad_ct.X
            tot = np.asarray(src.sum(axis=0)).ravel()
            keep_idx = np.sort(np.argsort(tot)[::-1][:args.max_genes])
            ad_ct = ad_ct[:, keep_idx].copy()
        cond = ad_ct.obs["condition"].to_numpy()
        n_ref = int((cond == args.ref_label).sum()); n_test = int((cond == args.test_label).sum())

        for m in methods:
            t0 = time.time()
            try:
                if m in ("duet", "scpyde"):
                    r = run_duet(ad_ct, ct, sample_col=args.sample_col, ref_label=args.ref_label,
                                 test_label=args.test_label, n_ref=n_ref, n_test=n_test,
                                 min_detect_frac=args.duet_min_detect_frac,
                                 linear_logfc=args.duet_linear_logfc)
                elif m == "diffxpy":
                    r = run_diffxpy(ad_ct, ct, counts_layer=args.counts_layer, condition=cond,
                                    ref_label=args.ref_label, test_label=args.test_label,
                                    n_ref=n_ref, n_test=n_test, min_cells=args.min_cells,
                                    max_cells_per_group=args.diffxpy_max_cells_per_group,
                                    max_genes=args.diffxpy_max_genes, nproc=args.diffxpy_nproc,
                                    seed=args.seed)
                elif m == "deseq2":
                    r = run_deseq2(ad_ct, ct, counts_layer=args.counts_layer, sample_col=args.source_col,
                                   condition=cond, ref_label=args.ref_label, test_label=args.test_label,
                                   n_ref=n_ref, n_test=n_test, min_samples_per_group=args.min_samples_per_group,
                                   min_gene_counts=args.min_gene_counts, min_cells_per_sample=args.min_cells_per_sample)
                elif m == "wilcoxon":
                    r = run_wilcoxon(ad_ct, ct, ref_label=args.ref_label, test_label=args.test_label,
                                     n_ref=n_ref, n_test=n_test)
                else:
                    _log(f"  unknown method '{m}', skipping"); continue
                n_tested = int((r["tested"] == True).sum())  # noqa: E712
                _log(f"  {m}: {n_tested} genes in {time.time()-t0:.0f}s")
                all_results.append(r)
            except Exception as e:
                _log(f"  {m}: FAILED ({type(e).__name__}: {str(e)[:160]}) -> recorded as failed")
                all_results.append(_empty(m, ct, n_ref, n_test, f"failed:{type(e).__name__}"))
        del ad_ct
        import gc; gc.collect()

    full = pd.concat(all_results, ignore_index=True)
    # per-method files
    for m in full["method"].unique():
        p = out_dir / f"de_{m}_all_celltypes.csv"
        full[full.method == m][HARMON].to_csv(p, index=False)
        _log(f"wrote {p}")

    _log("building comparison ...")
    build_comparison(full, fdr_thr=args.fdr, top_k=args.top_k, out_dir=out_dir)
    _log(f"DONE. Outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
