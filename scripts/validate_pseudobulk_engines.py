#!/usr/bin/env python
"""
validate_pseudobulk_engines.py -- is DUET's pseudobulk arm right?

Per dataset and cell type, on the raw integer UMI counts:
  1. the shipped wrapper, duet.pseudobulk.run_pseudobulk;
  2. an independent path -- counts summed with a pandas groupby over
     (donor, condition), then PyDESeq2 called directly;
  3. R DESeq2 (scripts/pseudobulk_deseq2.R) on the matrix from step 2.

The design is chosen from the data, not assumed: ~donor + condition only when every
donor contributes cells to both conditions (Kang), ~condition when every donor sits in
one condition (Crowell, pancreas); anything in between is reported and skipped.

Tolerances fixed in advance:
  wrapper vs direct PyDESeq2  identical: aggregated counts equal exactly, and
                              |log2FC|, |-log10 p| differences <= 1e-8;
  PyDESeq2 vs R DESeq2        different dispersion-fitting code, so agreement is
                              reported (Pearson log2FC, Spearman -log10 p, largest
                              differences, Jaccard of padj < 0.05), not assumed.

Output: evidence/pseudobulk_engine_agreement.csv
Usage:  python scripts/validate_pseudobulk_engines.py [--datasets kang,crowell,pancreas]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.background(1)

import argparse                                            # noqa: E402
import os                                                  # noqa: E402
import shutil                                              # noqa: E402
import subprocess                                          # noqa: E402
import tempfile                                            # noqa: E402
import warnings                                            # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
from scipy.stats import pearsonr, spearmanr                # noqa: E402

HERE = Path(__file__).resolve().parent.parent
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(HERE))
from donor_swap_partitions import DATASETS                 # noqa: E402

RSCRIPT = (os.environ.get("RSCRIPT") or shutil.which("Rscript")
           or r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe")
TEST = {"kang": "stim", "crowell": "LPS", "pancreas": "pancreas"}
MIN_CELLS, MIN_COUNTS = 10, 10
warnings.filterwarnings("ignore")


def independent_pseudobulk(A, donor_col, cond_col, ref, test):
    """Sum raw counts per (donor, condition) without touching duet's code."""
    import scipy.sparse as sp
    obs = A.obs[[donor_col, cond_col]].astype(str)
    key = obs[donor_col] + "|" + obs[cond_col]
    n = key.value_counts()
    key = key.where(key.map(n) >= MIN_CELLS)
    codes, uniq = pd.factorize(key)
    ok = codes >= 0
    G = sp.csr_matrix((np.ones(ok.sum()), (codes[ok], np.flatnonzero(ok))), shape=(len(uniq), A.n_obs))
    C = A.layers["counts"]
    M = np.asarray((G @ C).todense()).astype(np.int64)
    counts = pd.DataFrame(M, index=uniq, columns=A.var_names.astype(str))
    meta = pd.DataFrame({"donor": [u.split("|")[0] for u in uniq],
                         "condition": [u.split("|")[1] for u in uniq]}, index=uniq)
    meta = meta[meta.condition.isin([ref, test])]
    return counts.loc[meta.index], meta


def pydeseq2_direct(counts, meta, design, ref, test):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    counts = counts.loc[:, counts.sum(axis=0) >= MIN_COUNTS]
    meta = meta.copy()
    meta["condition"] = pd.Categorical(meta["condition"], categories=[ref, test])
    dds = DeseqDataSet(counts=counts, metadata=meta, design=design, quiet=True, n_cpus=1)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", test, ref], quiet=True, n_cpus=1)
    st.summary()
    r = st.results_df
    return pd.DataFrame({"log2FC": r["log2FoldChange"], "pvalue": r["pvalue"], "padj": r["padj"]},
                        index=r.index.astype(str)), counts


def r_deseq2(counts, meta, design, ref, test):
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        counts.T.to_csv(t / "counts.csv")
        meta.to_csv(t / "coldata.csv")
        r = subprocess.run([RSCRIPT, str(HERE / "scripts" / "pseudobulk_deseq2.R"), str(t / "counts.csv"),
                            str(t / "coldata.csv"), design, ref, test, str(t / "out.csv")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-800:])
        d = pd.read_csv(t / "out.csv")
    return d.set_index(d["gene"].astype(str))[["log2FC", "pvalue", "padj"]]


def nlp(p):
    return -np.log10(np.clip(np.asarray(p, float), 1e-300, 1))


def compare(a: pd.DataFrame, b: pd.DataFrame, tag: str) -> dict:
    g = a.index.intersection(b.index)
    ok = g[a.loc[g, "pvalue"].notna().to_numpy() & b.loc[g, "pvalue"].notna().to_numpy()]
    sa = set(ok[(a.loc[ok, "padj"] < 0.05).to_numpy()])
    sb = set(ok[(b.loc[ok, "padj"] < 0.05).to_numpy()])
    return {f"{tag}_n_genes": len(ok),
            f"{tag}_log2fc_pearson": pearsonr(a.loc[ok, "log2FC"], b.loc[ok, "log2FC"]).statistic,
            f"{tag}_log2fc_max_abs_diff": float((a.loc[ok, "log2FC"] - b.loc[ok, "log2FC"]).abs().max()),
            f"{tag}_nlp_spearman": spearmanr(nlp(a.loc[ok, "pvalue"]), nlp(b.loc[ok, "pvalue"])).statistic,
            f"{tag}_nlp_max_abs_diff": float(np.abs(nlp(a.loc[ok, "pvalue"]) - nlp(b.loc[ok, "pvalue"])).max()),
            f"{tag}_sig_a": len(sa), f"{tag}_sig_b": len(sb),
            f"{tag}_sig_jaccard": len(sa & sb) / max(len(sa | sb), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="kang,crowell,pancreas")
    ap.add_argument("--out", default=str(HERE / "evidence" / "pseudobulk_engine_agreement.csv"))
    a = ap.parse_args()
    from duet.pseudobulk import aggregate_pseudobulk, run_pseudobulk
    rows = []
    for name in a.datasets.split(","):
        cfg = DATASETS[name]
        ref, test = cfg["ref"], TEST[name]
        for ct in cfg["celltypes"]:
            A = _runtime.load_rows(cfg["h5ad"], lambda o: (o["celltype"].astype(str) == ct).to_numpy())
            A.obs[cfg["cond"]] = A.obs[cfg["cond"]].astype(str)
            both = A.obs.groupby(cfg["donor"], observed=True)[cfg["cond"]].nunique()
            if (both == 2).all():
                design, paired = "~donor + condition", True
            elif (both == 1).all():
                design, paired = "~condition", False
            else:
                rows.append(dict(dataset=name, celltype=ct, design="mixed -- skipped"))
                continue
            # exact aggregation check
            pb_w, meta_w = aggregate_pseudobulk(A, donor_col=cfg["donor"], condition_col=cfg["cond"],
                                                ref_label=ref, test_label=test, counts_layer="counts",
                                                min_cells_per_donor=MIN_CELLS)
            pb_i, meta_i = independent_pseudobulk(A, cfg["donor"], cfg["cond"], ref, test)
            key_w = meta_w["donor"].astype(str) + "|" + meta_w["condition"].astype(str)
            pb_w.index = key_w.to_numpy()
            same_counts = pb_w.shape == pb_i.shape and bool((pb_w.loc[pb_i.index].to_numpy() == pb_i.to_numpy()).all())
            w = run_pseudobulk(A, donor_col=cfg["donor"], condition_col=cfg["cond"], ref_label=ref,
                               test_label=test, counts_layer="counts", min_cells_per_donor=MIN_CELLS,
                               min_gene_counts=MIN_COUNTS, paired=paired, n_cpus=1)
            w = w[w.pb_tested].set_index("gene")
            w = w.rename(columns={"pb_log2FC": "log2FC", "pb_pvalue": "pvalue", "pb_padj": "padj"})
            d, counts_f = pydeseq2_direct(pb_i, meta_i, design, ref, test)
            r = r_deseq2(counts_f, meta_i, design, ref, test)
            row = dict(dataset=name, celltype=ct, design=design, n_samples=len(meta_i),
                       n_ref=int((meta_i.condition == ref).sum()), n_test=int((meta_i.condition == test).sum()),
                       aggregation_identical=same_counts)
            row.update(compare(w, d, "wrapper_vs_pydeseq2"))
            row.update(compare(d, r, "pydeseq2_vs_R"))
            row["wrapper_identical_within_1e-8"] = (row["wrapper_vs_pydeseq2_log2fc_max_abs_diff"] <= 1e-8
                                                    and row["wrapper_vs_pydeseq2_nlp_max_abs_diff"] <= 1e-8)
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[pb-check] wrote {a.out}")


if __name__ == "__main__":
    main()
