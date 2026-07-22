#!/usr/bin/env python
"""
Harmonize the R MAST results (mast_all_celltypes_pancreas.csv) into the comparison
as a 5th method, then rebuild the comparison from all 5 per-method CSVs.
MAST's only effect size is `coef` (natural-log mean difference) -> log2 via /ln2;
it has no group means, so (unlike DUET) it stays on the COMPRESSED log-mean-diff
scale. Restricted to the cell types the other methods used.
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_de_comparison as R

RESULTS = HERE / "results"
LN2 = math.log(2.0)

common_ct = list(pd.read_csv(RESULTS / "de_diffxpy_all_celltypes.csv")["celltype"].dropna().unique())
print("common cell types:", common_ct)

m = pd.read_csv(RESULTS / "mast_all_celltypes_pancreas.csv", low_memory=False)
print(f"MAST rows total={len(m)}; cell types={sorted(m['celltype'].dropna().unique())}")
m = m[m["celltype"].isin(common_ct)].copy()
tested = m["coef"].notna() & pd.to_numeric(m["Pr(>Chisq)"], errors="coerce").notna()
mast = pd.DataFrame({
    "method": "MAST", "celltype": m["celltype"].astype(str), "comparison": R.COMPARISON,
    "gene": m["primerid"].astype(str),
    "pvalue": pd.to_numeric(m["Pr(>Chisq)"], errors="coerce"),
    "fdr": pd.to_numeric(m["fdr"], errors="coerce"),
    "log2fc": pd.to_numeric(m["coef"], errors="coerce") / LN2,   # ln mean-diff -> log2
    # NOT recoverable: the R MAST CSV carries no test statistic, so a p-value that
    # underflowed to 0.0 (1.2-2.8% of genes here) is floored at 323 and its rank
    # within the tail is lost. Only DUET and Wilcoxon can be repaired.
    "neglog10p": R._nlp_naive(pd.to_numeric(m["Pr(>Chisq)"], errors="coerce").to_numpy(float)),
    "stat": np.nan, "n_ref": np.nan, "n_test": np.nan,
    "tested": tested.values, "skip_reason": np.where(tested.values, "", "untested_or_na"),
})
mast[R.HARMON].to_csv(RESULTS / "de_MAST_all_celltypes.csv", index=False)
print(f"wrote de_MAST_all_celltypes.csv  (rows={len(mast)}, tested={int(tested.sum())}, "
      f"cell types restricted to {common_ct})")

# ---- same-data sanity check: MAST vs DUET ----
scp = pd.read_csv(RESULTS / "de_DUET_all_celltypes.csv", low_memory=False)
for ct in common_ct:
    a = mast[(mast.celltype == ct) & (mast.tested)][["gene", "log2fc", "neglog10p"]].rename(
        columns={"log2fc": "lfc_MAST", "neglog10p": "nlp_MAST"})
    b = scp[(scp.celltype == ct) & (scp.tested == True)][["gene", "log2fc", "neglog10p"]].rename(
        columns={"log2fc": "lfc_scP", "neglog10p": "nlp_scP"})
    j = a.merge(b, on="gene").dropna()
    if len(j) > 50:
        sign = float((np.sign(j.lfc_MAST) == np.sign(j.lfc_scP)).mean()) * 100
        lfc_r = j["lfc_MAST"].corr(j["lfc_scP"])
        sp = j["nlp_MAST"].corr(j["nlp_scP"], method="spearman")
        print(f"  [{ct}] MAST vs DUET: n={len(j)} sign_agree={sign:.1f}%  "
              f"log2fc Pearson={lfc_r:.3f}  -log10p Spearman={sp:.3f}")

# ---- rebuild comparison from all 5 ----
order = ["DUET", "MAST", "diffxpy", "DESeq2", "wilcoxon"]
parts = [pd.read_csv(RESULTS / f"de_{name}_all_celltypes.csv", low_memory=False) for name in order]
full = pd.concat(parts, ignore_index=True)
R.build_comparison(full, fdr_thr=0.05, top_k=100, out_dir=RESULTS)
print("rebuilt comparison_*.csv with 5 methods")
