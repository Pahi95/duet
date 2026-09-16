#!/usr/bin/env python
"""
null_permutations.py -- the label-permutation null repeated 100 times per cell type.

null_calibration.py evaluates one permutation per cell type. Here the condition label
is permuted across all cells of a cell type (the donor structure is not preserved) with
seeds 1..N, and DUET, Wilcoxon and DESeq2 pseudobulk are run on every permutation.
Per method and permutation: tested genes, BH discoveries, type I error at alpha = 0.05
on the method's own tested genes, KS statistic and median p. A matched-gene BH on the
genes all three methods tested is recorded as well.

Outputs: results/null/permutations/<dataset>__<celltype>.csv (one row per seed x method)
         results/null/permutation_summary.csv
Usage:   python scripts/null_permutations.py [--datasets kang,crowell] [--n-perm 100] [--workers 3]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.limit_threads(1)

import argparse                                            # noqa: E402
import os                                                  # noqa: E402
import time                                                # noqa: E402
import warnings                                            # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed   # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from donor_swap_partitions import DATASETS                # noqa: E402

ALPHA = 0.05
TEST = {"kang": "stim", "crowell": "LPS"}


def run_celltype(name: str, celltype: str, n_perm: int, outdir: str) -> str:
    warnings.filterwarnings("ignore")
    _runtime.lower_priority()
    import run_de_comparison as R
    from null_calibration import summarize, _bh
    cfg = DATASETS[name]
    ref, test = cfg["ref"], TEST[name]
    dest = Path(outdir) / f"{name}__{celltype.replace(' ', '_').replace('.', '')}.csv"
    if dest.is_file():
        return f"[{name}/{celltype}] already done"
    t0 = time.time()
    adm = _runtime.load_rows(cfg["h5ad"], lambda o: (o["celltype"].astype(str) == celltype).to_numpy())
    real = np.where(adm.obs[cfg["cond"]].astype(str) == ref, ref, test)
    donors = adm.obs[cfg["donor"]].astype(str).to_numpy()
    ct_index = DATASETS[name]["celltypes"].index(celltype)
    rows = []
    for seed in range(1, n_perm + 1):
        rng = np.random.default_rng([seed, ct_index])
        c = rng.permutation(real)
        sub = adm.copy()
        sub.obs["condition"] = c
        sub.obs["Sample"] = c
        sub.obs["_donor"] = donors
        nr, nt = int((c == ref).sum()), int((c == test).sum())
        res = {
            "DUET": R.run_duet(sub, celltype, sample_col="Sample", ref_label=ref, test_label=test,
                               n_ref=nr, n_test=nt),
            "wilcoxon": R.run_wilcoxon(sub, celltype, ref_label=ref, test_label=test, n_ref=nr, n_test=nt),
            "DESeq2-pseudobulk": R.run_deseq2(sub, celltype, counts_layer="counts", sample_col="_donor",
                                              condition=c, ref_label=ref, test_label=test, n_ref=nr,
                                              n_test=nt, min_samples_per_group=2, min_gene_counts=10,
                                              min_cells_per_sample=10),
        }
        tested = {m: r[(r["tested"] == True) & r["pvalue"].notna()].set_index("gene")    # noqa: E712
                  for m, r in res.items()}
        common = sorted(set.intersection(*(set(t.index) for t in tested.values())))
        for m, r in res.items():
            s = summarize(r, m, celltype, "permutation", name)
            p = tested[m].loc[common, "pvalue"].to_numpy(float)
            s.update(seed=seed, n_common=len(common),
                     n_sig_matched=int((_bh(p) < ALPHA).sum()) if len(p) else 0,
                     type1_matched=round(float((p < ALPHA).mean()), 4) if len(p) else np.nan)
            rows.append(s)
        del sub
    df = pd.DataFrame(rows)
    tmp = dest.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    os.replace(tmp, dest)
    return f"[{name}/{celltype}] {n_perm} permutations, {adm.n_obs} cells, {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="kang,crowell")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default=str(HERE / "results" / "null" / "permutations"))
    a = ap.parse_args()
    _runtime.lower_priority()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(n, ct) for n in a.datasets.split(",") for ct in DATASETS[n]["celltypes"]]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_celltype, n, ct, a.n_perm, str(out)) for n, ct in tasks]
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print(f"[permutations] FAIL {type(e).__name__}: {e}", flush=True)
    D = pd.concat([pd.read_csv(f) for f in sorted(out.glob("*.csv"))], ignore_index=True)
    g = D.groupby(["dataset", "celltype", "method"], sort=False)
    S = g.agg(n_perm=("seed", "nunique"), n_tested_median=("n_tested", "median"),
              type1_mean=("type1_at_alpha05", "mean"), type1_min=("type1_at_alpha05", "min"),
              type1_max=("type1_at_alpha05", "max"), type1_matched_mean=("type1_matched", "mean"),
              median_p_mean=("median_p", "mean"),
              perms_with_any_discovery=("n_sig_fdr05", lambda x: int((x > 0).sum())),
              discoveries_max=("n_sig_fdr05", "max")).reset_index()
    S.to_csv(out.parent / "permutation_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
