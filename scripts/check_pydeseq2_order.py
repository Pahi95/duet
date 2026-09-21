#!/usr/bin/env python
"""
check_pydeseq2_order.py -- does the paired pseudobulk fit depend on sample order?

For each Kang cell type the pseudobulk matrix (raw counts summed per donor x
condition, >= 10 cells per sample, genes with >= 10 total counts) is fitted with
~donor + condition in three sample orders -- the order produced by the
aggregation, its reverse, and a random permutation (seed 0) -- once with PyDESeq2
called directly and once with R DESeq2 on the identical matrix.

Before every fit:
  * the rows of the count matrix and the rows of the metadata must be the same
    samples in the same order (asserted, not assumed);
  * donor and condition are categorical with fixed levels: donors sorted,
    condition levels (ctrl, stim); the reference is ctrl and the contrast is
    stim vs ctrl.

Outputs (evidence/pydeseq2_order/):
  <celltype>__genes.csv   one row per gene: log2FC, pvalue, padj for every engine
                          and order, plus the R-side independent-filtering and
                          Cook's-outlier flags
  order_summary.csv       per cell type, engine and order against the original
                          order: largest differences and significant-set changes
  engine_discordance.csv  per cell type, genes significant (padj < 0.05) in only
                          one engine, with their padj values, directions and the
                          reason the sets differ
  genewise_dispersion.csv per cell type, the PyDESeq2 dispersion step that differs
                          between orders

Usage:  python scripts/check_pydeseq2_order.py [--celltypes "CD14+ Monocytes,..."]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.background(1)

import argparse                                            # noqa: E402
import warnings                                            # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from donor_swap_partitions import DATASETS                 # noqa: E402
from validate_pseudobulk_engines import (independent_pseudobulk, r_deseq2,   # noqa: E402
                                         MIN_COUNTS)

warnings.filterwarnings("ignore")
REF, TEST = "ctrl", "stim"
DESIGN = "~donor + condition"
ALPHA = 0.05
NEAR = (0.01, 0.10)          # "close to the 0.05 boundary": both padj inside this band


def prepare(pb: pd.DataFrame, meta: pd.DataFrame, order: list[str]):
    """Reorder counts and metadata together and fix the factor levels."""
    c = pb.loc[order]
    m = meta.loc[order].copy()
    assert list(c.index) == list(m.index), "count rows and metadata rows differ"
    m["donor"] = pd.Categorical(m["donor"].astype(str), categories=sorted(meta["donor"].astype(str).unique()))
    m["condition"] = pd.Categorical(m["condition"].astype(str), categories=[REF, TEST])
    assert not m["donor"].isna().any() and not m["condition"].isna().any()
    return c, m


def fit_py(c, m):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    dds = DeseqDataSet(counts=c, metadata=m, design=DESIGN, quiet=True, n_cpus=1)
    dds.deseq2()
    cols = list(dds.obsm["design_matrix"].columns)
    assert cols[-1] == f"condition[T.{TEST}]", cols           # stim vs ctrl, ctrl is the reference
    st = DeseqStats(dds, contrast=["condition", TEST, REF], quiet=True, n_cpus=1)
    st.summary()
    r = st.results_df
    out = pd.DataFrame({"log2FC": r["log2FoldChange"], "pvalue": r["pvalue"], "padj": r["padj"]},
                       index=r.index.astype(str))
    disp = dds.var[["genewise_dispersions", "fitted_dispersions", "dispersions"]].copy()
    disp["genewise_converged"] = dds.var["_genewise_converged"].astype(bool)
    disp.attrs["trend"] = tuple(float(x) for x in dds.uns["trend_coeffs"])
    return out, disp


def fit_r(c, m):
    # R receives the samples in this order; its coldata is re-read and re-factored
    d = r_deseq2(c, m.astype({"donor": str, "condition": str}), DESIGN, REF, TEST)
    return d


def diff(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    g = a.index.intersection(b.index)
    a, b = a.loc[g], b.loc[g]
    nlp = lambda p: -np.log10(np.clip(p.astype(float), 1e-300, 1))
    sa, sb = a["padj"] < ALPHA, b["padj"] < ALPHA
    return {"n_genes": len(g),
            "max_abs_dlog2FC": float((a.log2FC - b.log2FC).abs().max()),
            "max_abs_dneglog10p": float((nlp(a.pvalue) - nlp(b.pvalue)).abs().max()),
            "max_abs_dpadj": float((a.padj - b.padj).abs().max()),
            "n_sig_original": int(sa.sum()), "n_sig_this": int(sb.sum()),
            "n_sig_changed": int((sa != sb).sum()),
            "identical": bool(np.allclose(a[["log2FC", "pvalue"]].to_numpy(float),
                                          b[["log2FC", "pvalue"]].to_numpy(float),
                                          rtol=0, atol=0, equal_nan=True))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltypes", default=",".join(DATASETS["kang"]["celltypes"]))
    ap.add_argument("--out", default=str(HERE / "evidence" / "pydeseq2_order"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DATASETS["kang"]
    summ, disc, disp_rows = [], [], []
    for ct in a.celltypes.split(","):
        A = _runtime.load_rows(cfg["h5ad"], lambda o: (o["celltype"].astype(str) == ct).to_numpy())
        A.obs[cfg["cond"]] = A.obs[cfg["cond"]].astype(str)
        pb, meta = independent_pseudobulk(A, cfg["donor"], cfg["cond"], REF, TEST)
        pb = pb.loc[:, pb.sum(axis=0) >= MIN_COUNTS]
        orig = list(pb.index)
        orders = {"original": orig, "reversed": orig[::-1],
                  "random": list(np.random.default_rng(0).permutation(orig))}
        genes = pd.DataFrame(index=pb.columns.astype(str))
        res, disps = {}, {}
        for name, o in orders.items():
            c, m = prepare(pb, meta, o)
            py, dsp = fit_py(c, m)
            r = fit_r(c, m)
            res[("py", name)], res[("R", name)], disps[name] = py, r, dsp
            for eng, t in (("py", py), ("R", r)):
                for col in ("log2FC", "pvalue", "padj"):
                    genes[f"{eng}_{name}_{col}"] = t[col].reindex(genes.index)
            print(f"[order] {ct} {name}: PyDESeq2 {int((py.padj < ALPHA).sum())} sig, "
                  f"R {int((r.padj < ALPHA).sum())} sig", flush=True)
        r0 = res[("R", "original")]
        genes["R_filtered_independent"] = r0["pvalue"].reindex(genes.index).notna() & r0["padj"].reindex(genes.index).isna()
        genes["R_cooks_outlier"] = r0["pvalue"].reindex(genes.index).isna() & r0["log2FC"].reindex(genes.index).notna()
        genes.index.name = "gene"
        genes.to_csv(out / f"{ct.replace(' ', '_').replace('+', 'p')}__genes.csv")

        for eng in ("py", "R"):
            for name in ("reversed", "random"):
                row = dict(celltype=ct, engine="PyDESeq2" if eng == "py" else "R DESeq2",
                           order=name, design=DESIGN, reference=REF, contrast=f"{TEST} vs {REF}",
                           n_samples=len(orig))
                row.update(diff(res[(eng, "original")], res[(eng, name)]))
                summ.append(row)

        # the shipped wrapper with engine="R", on the cells in a shuffled order, against
        # R DESeq2 called directly on the original order
        from duet.pseudobulk import run_pseudobulk
        B = A[np.random.default_rng(1).permutation(A.n_obs)].copy()
        w = run_pseudobulk(B, donor_col=cfg["donor"], condition_col=cfg["cond"], ref_label=REF,
                           test_label=TEST, min_cells_per_donor=10, min_gene_counts=MIN_COUNTS,
                           paired=True, engine="R")
        w = w[w.pb_tested].set_index("gene").rename(
            columns={"pb_log2FC": "log2FC", "pb_pvalue": "pvalue", "pb_padj": "padj"})
        row = dict(celltype=ct, engine="DUET wrapper, engine='R', cells shuffled", order="vs R original",
                   design=DESIGN, reference=REF, contrast=f"{TEST} vs {REF}", n_samples=len(orig))
        row.update(diff(res[("R", "original")], w))
        summ.append(row)

        # PyDESeq2 against R, original order
        py0 = res[("py", "original")]
        g = py0.index.intersection(r0.index)
        sp, sr = py0.loc[g, "padj"] < ALPHA, r0.loc[g, "padj"] < ALPHA
        # genes that change call between PyDESeq2 orders
        unstable = set()
        for name in ("reversed", "random"):
            o = res[("py", name)].loc[g, "padj"] < ALPHA
            unstable |= set(g[(o != sp).to_numpy()])
        for gene in g[(sp != sr).to_numpy()]:
            pa, ra = py0.loc[gene], r0.loc[gene]
            if pd.isna(pa.pvalue):
                reason = "PyDESeq2 Cook's outlier (p NA)"
            elif pd.isna(pa.padj):
                reason = "PyDESeq2 independent filtering (padj NA)"
            elif pd.isna(ra.padj) and pd.notna(ra.pvalue):
                reason = "R independent filtering (padj NA)"
            elif pd.isna(ra.pvalue):
                reason = "R Cook's outlier (p NA)"
            elif NEAR[0] <= pa.padj <= NEAR[1] and NEAR[0] <= ra.padj <= NEAR[1]:
                reason = "both padj near 0.05"
            else:
                reason = "dispersion trend differs"
            disc.append(dict(celltype=ct, gene=gene, sig_in="PyDESeq2" if sp[gene] else "R DESeq2",
                             py_padj=pa.padj, R_padj=ra.padj, py_log2FC=pa.log2FC, R_log2FC=ra.log2FC,
                             same_direction=bool(np.sign(pa.log2FC) == np.sign(ra.log2FC)),
                             py_dispersion=disps["original"].loc[gene, "dispersions"],
                             py_order_unstable=gene in unstable, reason=reason))

        d0 = disps["original"]
        for name in ("reversed", "random"):
            d1 = disps[name].loc[d0.index]
            rel = lambda col: (d0[col] - d1[col]).abs() / d0[col].abs()
            disp_rows.append(dict(
                celltype=ct, order=name,
                genewise_rel_diff_gt_1e3=int((rel("genewise_dispersions") > 1e-3).sum()),
                genewise_nonconverged_original=int((~d0.genewise_converged).sum()),
                genewise_nonconverged_this=int((~d1.genewise_converged).sum()),
                genewise_at_floor_original=int((d0.genewise_dispersions < 1e-7).sum()),
                trend_a0_original=d0.attrs["trend"][0], trend_a1_original=d0.attrs["trend"][1],
                trend_a0_this=disps[name].attrs["trend"][0], trend_a1_this=disps[name].attrs["trend"][1],
                fitted_trend_max_rel_diff=float(rel("fitted_dispersions").max()),
                final_max_rel_diff=float(rel("dispersions").max())))

    pd.DataFrame(summ).to_csv(out / "order_summary.csv", index=False)
    pd.DataFrame(disc).to_csv(out / "engine_discordance.csv", index=False)
    pd.DataFrame(disp_rows).to_csv(out / "genewise_dispersion.csv", index=False)
    print(pd.DataFrame(summ).to_string())
    print(f"[order] wrote {out}")


if __name__ == "__main__":
    main()
