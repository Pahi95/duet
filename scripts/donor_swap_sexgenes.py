#!/usr/bin/env python
"""
donor_swap_sexgenes.py -- are the donor-swap false discoveries driven by sex-chromosome genes?

Donors of different sex placed in different arms make X/Y genes (XIST, RPS4Y1, DDX3Y ...)
look differential although no condition effect exists. This reruns every donor partition
of donor_swap_partitions.py with the same models and records, per split and method:
  * the discoveries with BH on all common genes (reproduces results/null/donor_swap),
  * how many of them lie on chromosome X or Y and which genes they are,
  * the discoveries when X/Y genes are removed before BH ("autosomal").
Models are fitted on all genes; only the multiple-testing family changes.

Chromosome annotation: results/annotation/sexchrom_genes_{homo_sapiens,mus_musculus}.tsv
(Ensembl release 116 REST, see ENSEMBL_RELEASE.txt); genes are matched by symbol and,
where the input has one, Ensembl gene ID.

Outputs: results/null/donor_swap_sex/<dataset>__<celltype>.csv, results/null/donor_swap_sex_summary.csv
Usage:   python scripts/donor_swap_sexgenes.py [--datasets kang,crowell,pancreas] [--workers 3]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.limit_threads(1)

import argparse                                            # noqa: E402
import os                                                  # noqa: E402
import tempfile                                            # noqa: E402
import time                                                # noqa: E402
import warnings                                            # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed   # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from donor_swap_partitions import DATASETS, MIN_CELLS_PER_DONOR, ALPHA, bh, unique_splits   # noqa: E402

SPECIES = {"kang": "homo_sapiens", "crowell": "mus_musculus", "pancreas": "homo_sapiens"}


def sex_genes(name: str, var: pd.DataFrame) -> set[str]:
    ann = pd.read_csv(HERE / "results" / "annotation" / f"sexchrom_genes_{SPECIES[name]}.tsv", sep="\t")
    ids = set(ann.ensembl_gene_id.dropna().astype(str))
    syms = {s.upper() for s in ann.external_gene_name.dropna().astype(str)}
    hit = pd.Series(False, index=var.index)
    hit |= pd.Index(var.index.astype(str)).str.upper().isin(syms)
    hit |= pd.Index(var.index.astype(str)).str.split(".").str[0].isin(ids)
    for col in ("ENSEMBL", "SYMBOL", "name"):
        if col in var:
            v = var[col].astype(str)
            hit |= (v.str.upper().isin(syms) | v.str.split(".").str[0].isin(ids)).to_numpy()
    return set(var.index[hit.to_numpy()].astype(str))


def run_split(sub, donor_col, arm_a, arm_b):
    from duet import run_duet
    from duet.pseudobulk import run_pseudobulk
    import run_de_comparison as R
    d = sub.obs[donor_col].astype(str)
    A = sub[d.isin(arm_a + arm_b).to_numpy()].copy()
    A.obs["condition"] = np.where(A.obs[donor_col].astype(str).isin(arm_a), "armA", "armB")
    A.obs["Sample"] = A.obs["condition"]
    n_a, n_b = int((A.obs.condition == "armA").sum()), int((A.obs.condition == "armB").sum())
    with tempfile.TemporaryDirectory() as tmp:
        out = run_duet(A, output_dir=tmp, celltype_col="celltype", sample_col="Sample", ref_label="armA",
                       test_label="armB", mast_compat=True, vectorized=True, memory_log=False,
                       output_name="d.csv")
        du = pd.read_csv(out, low_memory=False)
    du = du[du.tested == True].set_index("gene")                                   # noqa: E712
    wi = R.run_wilcoxon(A, "x", ref_label="armA", test_label="armB", n_ref=n_a, n_test=n_b).set_index("gene")
    pb = run_pseudobulk(A, donor_col=donor_col, condition_col="condition", ref_label="armA", test_label="armB",
                        counts_layer="counts", min_cells_per_donor=MIN_CELLS_PER_DONOR,
                        min_donors_per_group=2, n_cpus=1)
    pb = pb[pb.pb_tested & pb.pb_pvalue.notna()].set_index("gene")
    common = du.index.intersection(wi.index[wi.pvalue.notna()]).intersection(pb.index)
    p = {"DUET": du.loc[common, "pvalue"].to_numpy(float), "wilcoxon": wi.loc[common, "pvalue"].to_numpy(float),
         "DESeq2-pseudobulk": pb.loc[common, "pb_pvalue"].to_numpy(float)}
    same_dir = np.sign(pb.loc[common, "pb_log2FC"].to_numpy()) == np.sign(du.loc[common, "coef"].to_numpy())
    return common.astype(str), p, same_dir, n_a, n_b


def calls(p, same_dir, mask):
    """BH within `mask`; returns method -> boolean discoveries over the masked genes."""
    q = {m: bh(v[mask]) < ALPHA for m, v in p.items()}
    q["DUET+PB"] = q["DESeq2-pseudobulk"] & same_dir[mask]
    return q


def run_celltype(name: str, celltype: str, outdir: str) -> str:
    warnings.filterwarnings("ignore")
    _runtime.lower_priority()
    cfg = DATASETS[name]
    dest = Path(outdir) / f"{name}__{celltype.replace(' ', '_').replace('.', '')}.csv"
    if dest.is_file():
        return f"[{name}/{celltype}] already done"
    t0 = time.time()
    sub = _runtime.load_rows(cfg["h5ad"], lambda o: ((o["celltype"].astype(str) == celltype)
                                                     & (o[cfg["cond"]].astype(str) == cfg["ref"])).to_numpy())
    sexset = sex_genes(name, sub.var)
    counts = sub.obs[cfg["donor"]].astype(str).value_counts()
    donors = sorted(counts.index[counts >= MIN_CELLS_PER_DONOR])
    rows = []
    for i, (arm_a, arm_b) in enumerate(unique_splits(donors)):
        genes, p, same_dir, n_a, n_b = run_split(sub, cfg["donor"], arm_a, arm_b)
        is_sex = np.isin(genes, list(sexset))
        allg = calls(p, same_dir, np.ones(len(genes), bool))
        auto = calls(p, same_dir, ~is_sex)
        for m in allg:
            disc = genes[allg[m]]
            sex_disc = sorted(set(disc) & sexset)
            rows.append(dict(dataset=name, celltype=celltype, split=i + 1, arm_a=";".join(arm_a),
                             arm_b=";".join(arm_b), n_cells_a=n_a, n_cells_b=n_b, method=m,
                             n_tested=len(genes), n_tested_sexchrom=int(is_sex.sum()),
                             n_discoveries=int(allg[m].sum()), n_discoveries_sexchrom=len(sex_disc),
                             sexchrom_discoveries=";".join(sex_disc[:50]),
                             n_tested_autosomal=int((~is_sex).sum()),
                             n_discoveries_autosomal=int(auto[m].sum()),
                             rejection_pct=100 * allg[m].sum() / max(len(genes), 1),
                             rejection_pct_autosomal=100 * auto[m].sum() / max(int((~is_sex).sum()), 1)))
    df = pd.DataFrame(rows)
    tmp = dest.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    os.replace(tmp, dest)
    return (f"[{name}/{celltype}] {len(donors)} donors, {df.split.nunique()} splits, {sub.n_obs} cells, "
            f"{len(sexset)} X/Y genes in input, {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="kang,crowell,pancreas")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default=str(HERE / "results" / "null" / "donor_swap_sex"))
    a = ap.parse_args()
    _runtime.lower_priority()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(n, ct) for n in a.datasets.split(",") for ct in DATASETS[n]["celltypes"]]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_celltype, n, ct, str(out)) for n, ct in tasks]
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print(f"[donor-swap-sex] FAIL {type(e).__name__}: {e}", flush=True)
    D = pd.concat([pd.read_csv(f) for f in sorted(out.glob("*.csv"))], ignore_index=True)
    D["any"] = D.n_discoveries > 0
    D["any_autosomal"] = D.n_discoveries_autosomal > 0
    g = D.groupby(["dataset", "celltype", "method"], sort=False)
    S = g.agg(n_splits=("split", "nunique"), rejection_pct_mean=("rejection_pct", "mean"),
              rejection_pct_max=("rejection_pct", "max"),
              rejection_pct_autosomal_mean=("rejection_pct_autosomal", "mean"),
              rejection_pct_autosomal_max=("rejection_pct_autosomal", "max"),
              discoveries_sum=("n_discoveries", "sum"), sexchrom_discoveries_sum=("n_discoveries_sexchrom", "sum"),
              share_splits_any=("any", "mean"), share_splits_any_autosomal=("any_autosomal", "mean")).reset_index()
    S.to_csv(out.parent / "donor_swap_sex_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
