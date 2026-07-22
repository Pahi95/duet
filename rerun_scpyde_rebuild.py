#!/usr/bin/env python
"""
Re-run ONLY scPyDE (fast) with the updated engine, reuse the cached
diffxpy/DESeq2/Wilcoxon per-method CSVs, and rebuild the comparison + report.
Avoids re-running the ~25-min diffxpy step.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_de_comparison as R

RESULTS = HERE / "results"
H5AD = HERE / "PancreasIntegratedAnnotated.h5ad"

# keep a copy of the previous scPyDE log2FC to show the before/after
old = pd.read_csv(RESULTS / "de_DUET_all_celltypes.csv")[["celltype", "gene", "log2fc"]]
old = old.rename(columns={"log2fc": "log2fc_OLD"})

celltypes = list(pd.read_csv(RESULTS / "de_diffxpy_all_celltypes.csv")["celltype"].dropna().unique())
print("cell types:", celltypes, flush=True)

A = ad.read_h5ad(str(H5AD), backed="r")
obs = A.obs
samp = obs["Sample"].astype(str)
condition_all = np.where(samp.str.contains("Reference", case=False, na=False), "Reference", "pancreas")
ct_labels = obs["celltype"].astype(str).to_numpy()

rows = []
for ct in celltypes:
    mask = ct_labels == ct
    view = A[mask]
    adm = ad.AnnData(X=view.X, obs=view.obs.copy(), var=A.var.copy())
    adm.layers["counts"] = view.layers["counts"]
    adm.obs["condition"] = condition_all[mask]
    cond = adm.obs["condition"].to_numpy()
    n_ref = int((cond == "Reference").sum()); n_test = int((cond == "pancreas").sum())
    r = R.run_duet(adm, ct, sample_col="Sample", ref_label="Reference",
                     test_label="pancreas", n_ref=n_ref, n_test=n_test)
    print(f"  {ct}: scPyDE tested {int((r['tested'] == True).sum())} genes", flush=True)
    rows.append(r)
    del adm
scp = pd.concat(rows, ignore_index=True)
scp[R.HARMON].to_csv(RESULTS / "de_DUET_all_celltypes.csv", index=False)
print("wrote de_DUET_all_celltypes.csv (linear log2FC)", flush=True)

# ---- before/after magnitude vs Wilcoxon ----
wil = pd.read_csv(RESULTS / "de_wilcoxon_all_celltypes.csv")[["celltype", "gene", "log2fc"]]
wil = wil.rename(columns={"log2fc": "log2fc_wilcox"})
new = scp[scp["tested"] == True][["celltype", "gene", "log2fc"]].rename(columns={"log2fc": "log2fc_NEW"})
cmp = new.merge(old, on=["celltype", "gene"]).merge(wil, on=["celltype", "gene"]).dropna()
print("\n=== scPyDE |log2FC| vs Wilcoxon |log2FC| (median ratio; ~1.0 = matched) ===")
for ct in celltypes:
    s = cmp[cmp.celltype == ct]
    if len(s):
        r_old = (s.log2fc_OLD.abs() / s.log2fc_wilcox.abs().clip(1e-6)).median()
        r_new = (s.log2fc_NEW.abs() / s.log2fc_wilcox.abs().clip(1e-6)).median()
        cor = s.log2fc_NEW.corr(s.log2fc_wilcox)
        print(f"  {ct:18s} OLD ratio={r_old:.2f}  ->  NEW ratio={r_new:.2f}   (NEW vs Wilcoxon Pearson r={cor:.3f})")

# ---- rebuild the full comparison from the 4 per-method CSVs ----
parts = [scp]
for m in ["diffxpy", "DESeq2", "wilcoxon"]:
    parts.append(pd.read_csv(RESULTS / f"de_{m}_all_celltypes.csv"))
full = pd.concat(parts, ignore_index=True)
R.build_comparison(full, fdr_thr=0.05, top_k=100, out_dir=RESULTS)
print("\nrebuilt comparison_*.csv + comparison_summary.txt", flush=True)
