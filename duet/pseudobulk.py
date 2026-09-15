"""
Sample-level (pseudobulk) testing, and the combined caller.

DUET's hurdle test operates on cells. Across donors that is a ranking tool
rather than a calling tool: cells from one donor are not independent
observations, so the false discovery rate at a nominal 0.05 is not merely
inflated but unpredictable -- on a replicated ground-truth simulation it ranged
from 0.047 to 0.873 across draws generated identically. Aggregating counts to
one pseudobulk profile per donor and testing at the sample level controls it;
in the same experiments a pseudobulk arm never exceeded 5% on any of 90
donor-swap splits.

This module supplies that second arm, so that the recommendation the method
implies -- rank with the hurdle, call with pseudobulk -- can be followed without
leaving DUET:

    from duet import run_duet_pseudobulk

    out = run_duet_pseudobulk(
        adata, output_dir="results",
        celltype_col="celltype",
        condition_col="condition", ref_label="ctrl", test_label="stim",
        donor_col="patient",          # who each cell came from
        counts_layer="counts",        # raw counts; the hurdle uses X
    )

The result carries both arms per gene and a `call` column that combines them.

The statistical engine is PyDESeq2, deliberately: re-implementing a
negative-binomial GLM with dispersion shrinkage would add a second inference
path to validate, and the aggregation -- not the test -- is what this module
exists to get right.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm

__all__ = ["aggregate_pseudobulk", "run_pseudobulk", "combine_calls",
           "run_duet_pseudobulk"]

# a gene is called on the sample-level arm; the cell-level arm only ranks
CALL_SIGNIFICANT = "significant"
CALL_RANKED_ONLY = "ranked_only"
CALL_DIRECTION_CONFLICT = "direction_conflict"
CALL_NS = "ns"
CALL_UNTESTED = "not_tested"


def _neglog10_from_wald(pvalue, stat):
    """-log10 p that survives the float64 floor, recovered from the Wald z.

    A p-value below ~1e-308 is stored as exactly 0.0, which ties every such gene
    and destroys the ranking. The two-sided normal tail has a closed form, so the
    value can be recovered from the statistic whenever the p-value has
    underflowed. This mirrors what the hurdle path does with the chi-squared
    tail.
    """
    p = np.asarray(pvalue, dtype=float)
    z = np.asarray(stat, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    out[ok] = -np.log10(np.clip(p[ok], np.finfo(float).tiny, 1.0))
    under = ok & (p <= 0.0) & np.isfinite(z)
    if under.any():
        # log(2 * sf(|z|)) in log space, then to base 10
        lg = math.log(2.0) + norm.logsf(np.abs(z[under]))
        out[under] = -lg / math.log(10.0)
    return out


def aggregate_pseudobulk(
    adata,
    *,
    donor_col: str,
    condition_col: str,
    ref_label,
    test_label,
    counts_layer: str = "counts",
    min_cells_per_donor: int = 10,
):
    """Sum raw counts into one profile per (donor, condition).

    Returns ``(counts, metadata)`` as DataFrames indexed by pseudobulk sample.

    The unit MUST be donor x condition, not donor alone. In a paired design --
    every donor contributing both control and stimulated cells, as in Kang 2018
    -- grouping by donor alone merges the two conditions into a single profile
    and then takes the condition label from an arbitrary cell, which silently
    collapses every sample into one group and produces a comparison with no
    contrast in it. Designs where each donor belongs entirely to one condition
    are unaffected: the grouping is then identical.
    """
    if counts_layer not in adata.layers:
        raise ValueError(
            f"pseudobulk needs raw counts, but layer {counts_layer!r} is not "
            f"present (layers: {list(adata.layers)}). The hurdle path uses "
            f"log-normalised X; a negative-binomial model needs integers."
        )
    counts = adata.layers[counts_layer]
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)

    donors = adata.obs[donor_col].astype(str).to_numpy()
    cond = adata.obs[condition_col].astype(str).to_numpy()
    genes = adata.var_names.astype(str).to_numpy()

    rows, meta_rows = [], []
    # Sorted, not in order of first appearance: PyDESeq2's fit of a paired
    # ~donor + condition design depends on the sample order (on Kang CD4 T cells a
    # shuffle moved one gene's -log10 p from 13 to 34), so a fixed order keeps the
    # result independent of how the cells happen to be ordered.
    for d in sorted(pd.unique(donors)):
        for lab in (str(ref_label), str(test_label)):
            m = (donors == d) & (cond == lab)
            n = int(m.sum())
            if n < min_cells_per_donor:
                continue
            rows.append(np.asarray(counts[m].sum(axis=0)).ravel())
            # keep the name unique when one donor appears in both conditions
            name = d if int((donors == d).sum()) == n else f"{d}|{lab}"
            meta_rows.append((name, d, lab, n))

    if not rows:
        return (pd.DataFrame(columns=genes),
                pd.DataFrame(columns=["donor", "condition", "n_cells"]))

    pb = pd.DataFrame(np.rint(np.vstack(rows)).astype(np.int64),
                      index=[r[0] for r in meta_rows], columns=genes)
    meta = pd.DataFrame(
        {"donor": [r[1] for r in meta_rows],
         "condition": pd.Categorical([r[2] for r in meta_rows],
                                     categories=[str(ref_label), str(test_label)]),
         "n_cells": [r[3] for r in meta_rows]},
        index=pb.index)
    return pb, meta


def run_pseudobulk(
    adata,
    *,
    donor_col: str,
    condition_col: str,
    ref_label,
    test_label,
    counts_layer: str = "counts",
    celltype: str = "",
    min_cells_per_donor: int = 10,
    min_donors_per_group: int = 2,
    min_gene_counts: int = 10,
    paired: bool = False,
    n_cpus: int | None = None,
):
    """DESeq2 on pseudobulk profiles. One row per gene.

    `paired=True` fits ``~donor + condition`` instead of ``~condition``, which is
    the stronger model when every donor contributes to both arms. It is not the
    default because it is only estimable in that design.

    `n_cpus` is passed to PyDESeq2, which otherwise starts one worker process per
    CPU; set it to 1 when this runs inside parallel jobs.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as e:                      # pragma: no cover
        raise ImportError(
            "the pseudobulk arm needs PyDESeq2:  pip install pydeseq2"
        ) from e

    pb, meta = aggregate_pseudobulk(
        adata, donor_col=donor_col, condition_col=condition_col,
        ref_label=ref_label, test_label=test_label,
        counts_layer=counts_layer, min_cells_per_donor=min_cells_per_donor)

    n_ref = int((meta["condition"] == str(ref_label)).sum()) if len(meta) else 0
    n_test = int((meta["condition"] == str(test_label)).sum()) if len(meta) else 0

    def _empty(reason):
        return pd.DataFrame({
            "gene": adata.var_names.astype(str),
            "celltype": celltype,
            "pb_log2FC": np.nan, "pb_pvalue": np.nan, "pb_padj": np.nan,
            "pb_neglog10p": np.nan, "pb_stat": np.nan,
            "pb_tested": False, "pb_skip_reason": reason,
            "n_donors_ref": n_ref, "n_donors_test": n_test,
        })

    # Fail loudly rather than returning a table of NaN that reads as "nothing
    # was differential": too few samples means the question was not asked.
    if n_ref < min_donors_per_group or n_test < min_donors_per_group:
        return _empty(f"need>={min_donors_per_group} donors/group "
                      f"(ref={n_ref}, test={n_test})")

    keep = pb.sum(axis=0) >= min_gene_counts
    pb = pb.loc[:, keep]
    if pb.shape[1] == 0:
        return _empty("no_genes_after_count_filter")

    design = "~donor + condition" if paired else "~condition"
    if paired and meta["donor"].nunique() == len(meta):
        raise ValueError(
            "paired=True needs donors contributing to both conditions, but every "
            "pseudobulk sample comes from a different donor; the donor term "
            "would be collinear with the condition term.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=pb, metadata=meta, design=design, quiet=True, n_cpus=n_cpus)
        dds.deseq2()
        st = DeseqStats(dds, contrast=["condition", str(test_label), str(ref_label)],
                        quiet=True, n_cpus=n_cpus)
        st.summary()

    r = st.results_df.reset_index()
    gcol = "gene" if "gene" in r.columns else r.columns[0]
    res = pd.DataFrame({
        "gene": r[gcol].astype(str),
        "celltype": celltype,
        "pb_log2FC": r["log2FoldChange"].astype(float),
        "pb_pvalue": r["pvalue"].astype(float),
        "pb_padj": r["padj"].astype(float),
        "pb_neglog10p": _neglog10_from_wald(r["pvalue"].astype(float),
                                            r["stat"].astype(float)),
        "pb_stat": r["stat"].astype(float),
        "pb_tested": True,
        "pb_skip_reason": "",
        "n_donors_ref": n_ref,
        "n_donors_test": n_test,
    })
    # genes dropped by the count filter still get a row, so the two arms join cleanly
    missing = set(adata.var_names.astype(str)) - set(res["gene"])
    if missing:
        pad = _empty("below_min_gene_counts")
        pad = pad[pad["gene"].isin(missing)]
        res = pd.concat([res, pad], ignore_index=True)
    return res


def combine_calls(duet_df, pb_df, *, alpha: float = 0.05, require_same_direction: bool = True):
    """Join the cell-level and sample-level arms into one table.

    Adds `call`, the recommendation made operational:

    ``significant``        the sample-level arm supports the gene (pb_padj < alpha)
                           and, with ``require_same_direction``, its fold change has
                           the same sign as the hurdle's coefficient
    ``direction_conflict`` the sample-level arm is significant but the two arms
                           disagree on the direction -- not counted as a discovery
    ``ranked_only``        the hurdle ranks the gene highly, pseudobulk does not
    ``ns``                 neither arm

    The hurdle's own `fdr` is kept, but it never decides the call. The direction
    check is part of the rule stated in the manuscript: a gene called from opposite
    directions by the two arms is not evidence for either.
    """
    key = ["gene", "celltype"]
    for name, df in (("duet", duet_df), ("pseudobulk", pb_df)):
        miss = [k for k in key if k not in df.columns]
        if miss:
            raise ValueError(f"{name} table is missing {miss}")

    out = duet_df.merge(pb_df, on=key, how="outer", suffixes=("", "_pb"))

    pb_sig = out["pb_padj"].notna() & (out["pb_padj"] < alpha)
    duet_sig = out["fdr"].notna() & (out["fdr"] < alpha)
    tested = out.get("pb_tested", pd.Series(False, index=out.index)).fillna(False)

    coef = out["coef"] if "coef" in out.columns else out.get("coef_continuous")
    if require_same_direction and coef is not None:
        same_dir = np.sign(out["pb_log2FC"].to_numpy(float)) == np.sign(np.asarray(coef, dtype=float))
    else:
        same_dir = np.ones(len(out), dtype=bool)
    sig = pb_sig.to_numpy(bool) & same_dir

    call = np.where(sig, CALL_SIGNIFICANT,
                    np.where(pb_sig.to_numpy(bool), CALL_DIRECTION_CONFLICT,
                             np.where(duet_sig, CALL_RANKED_ONLY, CALL_NS)))
    out["call"] = np.where(tested.to_numpy(bool), call, CALL_UNTESTED)
    return out


def run_duet_pseudobulk(
    adata,
    output_dir: str = "results",
    *,
    celltype_col: str = "celltype",
    condition_col: str,
    ref_label,
    test_label,
    donor_col: str,
    counts_layer: str = "counts",
    alpha: float = 0.05,
    paired: bool = False,
    require_same_direction: bool = True,
    min_cells_per_donor: int = 10,
    min_donors_per_group: int = 2,
    min_gene_counts: int = 10,
    n_cpus: int | None = None,
    output_name: str = "duet_pseudobulk.csv",
    **duet_kwargs,
):
    """Run both arms and write one table carrying both.

    The hurdle supplies the ranking (`neglog10p`), pseudobulk supplies the call
    (`pb_padj`), and `call` combines them. Extra keyword arguments are passed
    through to :func:`duet.run_duet`.

    `donor_col` is the new requirement and the reason this could not be done
    before: DUET's own API has no notion of who a cell came from -- its
    `sample_col` is the condition column -- so it had no way to aggregate.
    """
    import os
    from pathlib import Path

    from .core import run_duet

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if donor_col not in adata.obs:
        raise ValueError(f"donor_col {donor_col!r} is not in adata.obs")
    if condition_col not in adata.obs:
        raise ValueError(f"condition_col {condition_col!r} is not in adata.obs")
    if donor_col == condition_col:
        raise ValueError(
            "donor_col and condition_col must differ: the donor is who the cell "
            "came from, the condition is what is being compared. Passing the "
            "same column collapses every pseudobulk sample into one group.")

    # ---- arm 1: the cell-level hurdle, unchanged ------------------------
    duet_path = run_duet(
        adata, output_dir=str(out_dir), celltype_col=celltype_col,
        sample_col=condition_col, condition_col=condition_col,
        ref_label=ref_label, test_label=test_label,
        output_name="_duet_cell_level.csv", **duet_kwargs)
    cell = pd.read_csv(duet_path, low_memory=False)

    # ---- arm 2: pseudobulk, per cell type -------------------------------
    ct_all = adata.obs[celltype_col].astype(str)
    frames = []
    for ct in pd.unique(ct_all):
        sub = adata[ct_all.to_numpy() == ct]
        frames.append(run_pseudobulk(
            sub, donor_col=donor_col, condition_col=condition_col,
            ref_label=ref_label, test_label=test_label,
            counts_layer=counts_layer, celltype=ct,
            min_cells_per_donor=min_cells_per_donor,
            min_donors_per_group=min_donors_per_group,
            min_gene_counts=min_gene_counts, paired=paired, n_cpus=n_cpus))
    pb = pd.concat(frames, ignore_index=True)

    combined = combine_calls(cell, pb, alpha=alpha, require_same_direction=require_same_direction)
    out = out_dir / output_name
    combined.to_csv(out, index=False)

    n_sig = int((combined["call"] == CALL_SIGNIFICANT).sum())
    n_rank = int((combined["call"] == CALL_RANKED_ONLY).sum())
    n_conf = int((combined["call"] == CALL_DIRECTION_CONFLICT).sum())
    print(f"[duet+pb] {out}")
    print(f"[duet+pb] {n_sig} significant (sample-level), "
          f"{n_rank} ranked_only (cell-level only), {n_conf} direction conflicts")
    try:
        os.unlink(duet_path)
    except OSError:
        pass
    return str(out)
