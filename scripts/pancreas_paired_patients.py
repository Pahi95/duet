#!/usr/bin/env python
"""
pancreas_paired_patients.py -- does the tumor-vs-reference pseudobulk depend on patients counted twice?

Eleven patients contributed a tumor sample and a tumor-adjacent non-tumor sample (Additional
file 1, column `patient`). The pipeline gives the adjacent sample the DonorID <patient>_N, so the
pseudobulk of results/pancreas_20260925/de treats the two samples of one patient as independent
(design ~condition). This script rebuilds exactly those pseudobulk samples (DonorID x condition,
>= 10 cells, genes with >= 10 summed counts) and refits them three ways:

  original      ~condition on every sample (reproduces de_DESeq2_all_celltypes.csv)
  drop_tumor    ~condition without the tumor samples of the paired patients, so every patient
                is in one condition only (the reference group, the smaller one, is kept whole)
  drop_normal   ~condition without the adjacent non-tumor samples of the paired patients

PyDESeq2, as in the original run. The patient-blocked limma-voom fit (duplicateCorrelation) is in
pancreas_paired_patients_limma.R and reads the matrices written here.

Input:   the PDAC_09_25 h5ad named in inputs/bench/pancreas_20260925/manifest.json,
         manuscript/DUET_additional_file_1_samples_v8.csv (patient of each DonorID)
Outputs: results/pancreas_20260925/paired_patients/{pb_<ct>.csv, meta_<ct>.csv, de_<design>.csv,
         summary.csv}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
from _runtime import load_rows, read_obs                     # noqa: E402

OUT = HERE / "results" / "pancreas_20260925" / "paired_patients"
ORIG = HERE / "results" / "pancreas_20260925" / "de" / "de_DESeq2_all_celltypes.csv"
MANIFEST = HERE.parent / "manuscript" / "DUET_additional_file_1_samples_v8.csv"
CELLTYPES = ["Ductal cell", "Endothelial cell", "Stellate cell"]
REF, TEST = "Reference", "pancreas"
MIN_CELLS, MIN_GENE_COUNTS, ALPHA = 10, 10, 0.05


def pseudobulk(h5ad, obs, ct):
    mask = obs["celltype"].astype(str).to_numpy() == ct
    A = load_rows(h5ad, mask, layers=("counts",))
    X = A.layers["counts"]
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    cond = np.where(A.obs["Sample"].astype(str).str.contains(REF, case=False), REF, TEST)
    donor = A.obs["DonorID"].astype(str).to_numpy()
    rows, meta = [], []
    for d in sorted(pd.unique(donor)):
        for lab in (REF, TEST):
            m = (donor == d) & (cond == lab)
            if int(m.sum()) < MIN_CELLS:
                continue
            rows.append(np.asarray(X[m].sum(axis=0)).ravel())
            meta.append((d, lab, int(m.sum())))
    pb = pd.DataFrame(np.rint(np.vstack(rows)).astype(int), index=[d for d, _, _ in meta],
                      columns=A.var_names.astype(str))
    meta = pd.DataFrame(meta, columns=["donor", "condition", "n_cells"]).set_index("donor", drop=False)
    pb = pb.loc[:, pb.sum(axis=0) >= MIN_GENE_COUNTS]
    return pb, meta


def deseq(pb, meta):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    md = pd.DataFrame({"condition": pd.Categorical(meta["condition"], categories=[REF, TEST])},
                      index=meta.index)
    dds = DeseqDataSet(counts=pb, metadata=md, design="~condition", quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", TEST, REF], quiet=True)
    st.summary()
    r = st.results_df
    return pd.DataFrame({"gene": r.index.astype(str), "pvalue": r["pvalue"].to_numpy(float),
                         "fdr": r["padj"].to_numpy(float), "log2fc": r["log2FoldChange"].to_numpy(float)})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h5ad = json.loads((HERE.parent / "inputs" / "bench" / "pancreas_20260925" / "manifest.json")
                      .read_text(encoding="utf-8"))["source"]
    man = pd.read_csv(MANIFEST).drop_duplicates("library")
    patient = man.groupby("donor")["patient"].first()
    both = man.groupby("patient")["group"].nunique()
    paired = set(both[both > 1].index)
    obs = read_obs(h5ad)
    orig = pd.read_csv(ORIG)

    rows = []
    for ct in CELLTYPES:
        pb, meta = pseudobulk(h5ad, obs, ct)
        meta["patient"] = meta["donor"].map(patient).fillna(meta["donor"].str.replace(r"_N$", "", regex=True))
        npc = meta.groupby("patient")["condition"].nunique()
        meta["paired_in_celltype"] = meta["patient"].map(npc).eq(2)
        tag = ct.split()[0].lower()
        pb.to_csv(OUT / f"pb_{tag}.csv")
        meta.to_csv(OUT / f"meta_{tag}.csv", index=False)
        pairs = sorted(meta.loc[meta["paired_in_celltype"], "patient"].unique())
        print(f"{ct}: {len(meta)} samples ({(meta.condition == REF).sum()} ref), "
              f"{len(pairs)} patients in both conditions: {pairs}", flush=True)
        assert set(pairs) <= paired

        designs = {
            "original": meta.index,
            "drop_tumor": meta.index[~(meta["paired_in_celltype"] & (meta["condition"] == TEST))],
            "drop_normal": meta.index[~(meta["paired_in_celltype"] & (meta["condition"] == REF))],
        }
        o = orig[(orig.celltype == ct) & orig.tested].set_index("gene")
        o_sig = set(o.index[o.fdr < ALPHA])
        base = None
        for name, idx in designs.items():
            sub = pb.loc[idx]
            sub = sub.loc[:, sub.sum(axis=0) >= MIN_GENE_COUNTS]
            r = deseq(sub, meta.loc[idx])
            r.insert(0, "celltype", ct)
            r.insert(0, "design", name)
            r.to_csv(OUT / f"de_{name}_{tag}.csv", index=False)
            sig = set(r.gene[r.fdr < ALPHA])
            if name == "original":
                base = r.set_index("gene")
            common = base.index.intersection(r.set_index("gene").index)
            rr = r.set_index("gene").loc[common]
            bb = base.loc[common]
            b_sig = set(bb.index[bb.fdr < ALPHA])
            rows.append(dict(
                celltype=ct, design=name, n_samples=len(idx),
                n_ref=int((meta.loc[idx, "condition"] == REF).sum()),
                n_test=int((meta.loc[idx, "condition"] == TEST).sum()),
                n_paired_patients=len(pairs), n_genes=len(r), n_sig=len(sig),
                sig_same_as_published=sig == o_sig if name == "original" else None,
                jaccard_vs_original=len(sig & b_sig) / max(len(sig | b_sig), 1),
                sig_lost=len(b_sig - sig), sig_gained=len(sig - b_sig),
                log2fc_pearson_vs_original=float(np.corrcoef(rr.log2fc, bb.log2fc)[0, 1]),
                median_abs_log2fc_change=float(np.nanmedian(np.abs(rr.log2fc - bb.log2fc))),
            ))
            print(f"  {name:12s} samples={len(idx):3d} sig={len(sig):5d} "
                  f"J={rows[-1]['jaccard_vs_original']:.3f} r={rows[-1]['log2fc_pearson_vs_original']:.4f}",
                  flush=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(f"wrote {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
