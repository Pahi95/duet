#!/usr/bin/env python
"""
wald_t_check.py -- DESeq2 pseudobulk with a t instead of a normal reference for the Wald test.

With four donors per group the default Wald test (normal reference) exceeded the nominal
family-wise level in the negative-binomial global nulls, and a t reference removed the excess
(simulate_batch_nb_wald_t.R). This script repeats the comparison on
  partitions  the negative-control donor partitions of Table 6 / Table S16, split by split as in
              results/null/donor_swap_sex_full (Kang, Crowell) and
              results/pancreas_20260925/null/donor_swap_sex_full (pancreas). BH runs over the genes
              DUET, Wilcoxon and pseudobulk all tested, exactly as donor_swap_sexgenes.py; the
              PyDESeq2 p-values of that script are refitted alongside to check the reproduction.
  sims        the 100 muscat replicates (sensitivity and FDP at BH 0.05, ~condition, 4 vs 4).
Pseudobulk matrices are built as in run_pseudobulk / run_de_comparison.run_deseq2 (raw counts per
donor, >= 10 cells, genes with >= 10 summed counts) and fitted by deseq2_wald_t.R.

Outputs: results/wald_t/{partitions,sims}/... and results/wald_t/{partitions,sims}_summary.csv
Usage:   python scripts/wald_t_check.py [--part partitions|sims|both]   (RSCRIPT, R_LIBS_USER for R)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.limit_threads(1)

import argparse                                            # noqa: E402
import os                                                  # noqa: E402
import subprocess                                          # noqa: E402
import warnings                                            # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import scipy.sparse as sp                                  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "wald_t"
ALPHA, MIN_CELLS, MIN_GENE_COUNTS = 0.05, 10, 10
SPLITS = {"kang": HERE / "results" / "null" / "donor_swap_sex_full",
          "crowell": HERE / "results" / "null" / "donor_swap_sex_full",
          "pancreas": HERE / "results" / "pancreas_20260925" / "null" / "donor_swap_sex_full"}
TRUE_DE = ("de", "dp", "dm", "db")


def bh(p):
    from donor_swap_partitions import bh as _bh
    return _bh(p)


def dump_pb(X, donor, cond, genes, key, folder):
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    rows, names, labs = [], [], []
    for d in sorted(pd.unique(donor)):
        m = donor == d
        if int(m.sum()) < MIN_CELLS:
            continue
        rows.append(np.asarray(X[m].sum(axis=0)).ravel())
        names.append(d)
        labs.append(cond[m][0])
    pb = pd.DataFrame(np.rint(np.vstack(rows)).astype(int), index=names, columns=genes)
    pb = pb.loc[:, pb.sum(axis=0) >= MIN_GENE_COUNTS]
    pb.T.to_csv(folder / f"counts_{key}.csv")
    pd.DataFrame({"condition": labs}, index=names).to_csv(folder / f"coldata_{key}.csv")


def run_r(folder, ref, test):
    (folder / "levels.txt").write_text(f"{ref},{test}\n")
    r = subprocess.run([os.environ.get("RSCRIPT", "Rscript"), str(HERE / "scripts" / "deseq2_wald_t.R"),
                        str(folder)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])


def read_out(folder, key):
    return pd.read_csv(folder / f"out_{key}.csv", dtype={"gene": str}, keep_default_na=False,
                       na_values={c: ["NA"] for c in ("pvalue_z", "padj_z", "pvalue_t", "padj_t",
                                                       "log2FoldChange")}).set_index("gene")


def partitions():
    from donor_swap_partitions import DATASETS
    from donor_swap_sexgenes import run_split
    rows = []
    for name in ("kang", "crowell", "pancreas"):
        cfg = DATASETS[name]
        for f in sorted(SPLITS[name].glob(f"{name}__*.csv")):
            S = pd.read_csv(f)
            ct = S.celltype.iloc[0]
            pbrow = S[S.method == "DESeq2-pseudobulk"].set_index("split")
            folder = OUT / "partitions" / f.stem
            folder.mkdir(parents=True, exist_ok=True)
            sub = _runtime.load_rows(cfg["h5ad"], lambda o: ((o["celltype"].astype(str) == ct)
                                     & (o[cfg["cond"]].astype(str) == cfg["ref"])).to_numpy())
            donor_all = sub.obs[cfg["donor"]].astype(str).to_numpy()
            common = {}
            for split, r in pbrow.iterrows():
                arm_a, arm_b = tuple(r.arm_a.split(";")), tuple(r.arm_b.split(";"))
                genes, p, _, _, _ = run_split(sub, cfg["donor"], arm_a, arm_b)
                py_disc = int(np.nansum(bh(p["DESeq2-pseudobulk"]) < ALPHA))
                common[split] = (pd.Index(genes), py_disc)
                keep = np.isin(donor_all, arm_a + arm_b)
                cond = np.where(np.isin(donor_all[keep], arm_a), "armA", "armB")
                dump_pb(sub.layers["counts"][keep], donor_all[keep], cond, sub.var_names.astype(str),
                        f"{split:03d}", folder)
            run_r(folder, "armA", "armB")
            for split, (genes, py_disc) in common.items():
                o = read_out(folder, f"{split:03d}")
                g = genes.intersection(o.index[o.pvalue_z.notna() & o.pvalue_t.notna()])
                qz, qt = bh(o.loc[g, "pvalue_z"]), bh(o.loc[g, "pvalue_t"])
                rows.append(dict(dataset=name, celltype=ct, split=split, n_tested_published=int(pbrow.loc[split, "n_tested"]),
                                 n_tested=len(g), disc_published=int(pbrow.loc[split, "n_discoveries"]),
                                 disc_pydeseq2_rerun=py_disc, disc_R_normal=int((qz < ALPHA).sum()),
                                 disc_R_t=int((qt < ALPHA).sum())))
            print(f"[partitions] {name}/{ct}: {len(common)} splits", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "partitions.csv", index=False)
    S = D.groupby(["dataset", "celltype"], sort=False).agg(
        splits=("split", "size"),
        any_published=("disc_published", lambda x: int((x > 0).sum())),
        any_pydeseq2_rerun=("disc_pydeseq2_rerun", lambda x: int((x > 0).sum())),
        any_R_normal=("disc_R_normal", lambda x: int((x > 0).sum())),
        any_R_t=("disc_R_t", lambda x: int((x > 0).sum())),
        mean_pct_R_normal=("disc_R_normal", "sum"), mean_pct_R_t=("disc_R_t", "sum"),
        n_tested_mean=("n_tested", "mean")).reset_index()
    S["mean_pct_R_normal"] = 100 * S.mean_pct_R_normal / (S.splits * S.n_tested_mean)
    S["mean_pct_R_t"] = 100 * S.mean_pct_R_t / (S.splits * S.n_tested_mean)
    S.to_csv(OUT / "partitions_summary.csv", index=False)
    print(S.to_string(index=False))


def sims():
    from evaluate_sim_replicates import build
    INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
    folder = OUT / "sims"
    folder.mkdir(parents=True, exist_ok=True)
    truth = {}
    for rep in range(1, 101):
        src = INPUTS / "data" / f"sim_rep{rep:02d}"
        key = f"{rep:03d}"
        gi = pd.read_csv(src / "gene_info.tsv", sep="\t").drop_duplicates("gene").set_index("gene")
        truth[key] = gi["category"].astype(str)
        if (folder / f"counts_{key}.csv").is_file():
            continue
        A = build(src)
        dump_pb(A.layers["counts"], A.obs["SourceFile"].astype(str).to_numpy(),
                A.obs["Sample"].astype(str).to_numpy(), A.var_names.astype(str), key, folder)
    run_r(folder, "A", "B")
    rows = []
    for key, cat in truth.items():
        o = read_out(folder, key)
        stored = pd.read_csv(HERE / "results" / "sim_reps100" / "calls" / f"sim_rep{int(key):02d}" / "deseq2.csv",
                             dtype={"gene": str}).set_index("gene")
        for label, padj in (("PyDESeq2 (published)", stored["fdr"]), ("R normal", o["padj_z"]), ("R t", o["padj_t"])):
            c = cat.reindex(padj.index)
            ok = c.notna()
            alt = c[ok].isin(TRUE_DE)
            called = (padj[ok] < ALPHA).fillna(False)
            tp, fp = int((called & alt).sum()), int((called & ~alt).sum())
            rows.append(dict(replicate=int(key), method=label, sensitivity=tp / max(int(alt.sum()), 1),
                             fdp=fp / max(tp + fp, 1), n_called=tp + fp))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "sims.csv", index=False)
    S = D.groupby("method", sort=False).agg(sensitivity=("sensitivity", "mean"), fdp=("fdp", "mean"),
                                            n_called=("n_called", "mean")).reset_index()
    S.to_csv(OUT / "sims_summary.csv", index=False)
    print(S.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="both", choices=["partitions", "sims", "both"])
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part in ("sims", "both"):
        sims()
    if a.part in ("partitions", "both"):
        partitions()


if __name__ == "__main__":
    main()
