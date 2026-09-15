#!/usr/bin/env python
"""
build_pancreas_corrected.py -- the pancreas input the revised analyses use.

Two defects in PancreasIntegratedAnnotated.h5ad make it unsuitable as it stands:

1. layers["counts"] is a copy of X, i.e. log1p(normalize_total(1e4)), not raw UMIs.
   The raw counts are recoverable exactly: counts = expm1(X) * total_counts / 1e4
   is integer-valued to within float32 rounding, sums to obs.total_counts and has
   obs.n_genes_by_counts non-zeros in every cell (checked below; the build aborts
   otherwise).

2. The 16 non-tumour files are not 16 samples. They are sequencing runs (FGC flow
   cells) of five HPAP donor libraries, and the same cell barcode appears in two
   to four of them with correlated but different depths. Loaded side by side they
   count one physical cell up to four times. Runs are merged here by (donor,
   barcode) by summing counts -- what Cell Ranger does with several sequencing runs
   of one library. The cell type of a merged cell is the majority label of its
   copies (ties: the copy with the most UMIs).

File -> library mapping: the user-supplied HPAP names in ASCII-sorted order map to
files 1..16; this ordering, and no other, makes every group of files that share
barcodes belong to a single donor (revision_MO/_work/hpap_barcode_overlap.py).

Outputs (the original object is not modified):
  inputs/PancreasCorrected.h5ad          X = log1p(CP10k), layers["counts"] = int32 UMIs
  evidence/pancreas_sample_table.csv     file -> donor / library / run, cells before and after merging
"""
from __future__ import annotations

import argparse
import os
import re
from datetime import date
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent.parent            # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
TARGET_SUM = 1e4

HPAP_NAMES = sorted("""
HPAP077_103411 HPAP-092_FGC2332_105153 HPAP-099_FGC2390_106838 HPAP-099_FGC2512_106838
HPAP-101_FGC2507_106840 HPAP-104_FGC2430_107485 HPAP-077_FGC2507_103411 HPAP-092_FGC2390_105153
HPAP-099_FGC2430_106838 HPAP-101_FGC2390_106840 HPAP-101_FGC2512_106840 HPAP-077_FGC2512_103411
HPAP-092_FGC2430_105153 HPAP-099_FGC2507_106838 HPAP-101_FGC2430_106840 HPAP-104_FGC2390_107485
""".split())
FILE_TO_HPAP = {str(i + 1): name for i, name in enumerate(HPAP_NAMES)}


def parse_hpap(name: str) -> tuple[str, str, str]:
    """'HPAP-099_FGC2390_106838' -> ('HPAP-099', '106838', 'FGC2390'); 'HPAP077_103411' has no run id."""
    m = re.match(r"HPAP-?(\d{3})_(?:(FGC\d+)_)?(\d+)$", name)
    if not m:
        raise ValueError(f"unrecognised HPAP name {name!r}")
    return f"HPAP-{m.group(1)}", m.group(3), m.group(2) or "run_unlabelled"


def recover_counts(X: sp.csr_matrix, total_counts: np.ndarray, chunk: int = 20000):
    """Invert log1p(normalize_total) row by row; return int32 CSR and the worst rounding error."""
    X = X.tocsr()
    out_data = np.empty(X.nnz, dtype=np.int32)
    worst = 0.0
    for start in range(0, X.shape[0], chunk):
        stop = min(start + chunk, X.shape[0])
        lo, hi = X.indptr[start], X.indptr[stop]
        rows = np.repeat(np.arange(start, stop), np.diff(X.indptr[start:stop + 1]))
        vals = np.expm1(X.data[lo:hi].astype(np.float64)) * (total_counts[rows] / TARGET_SUM)
        r = np.rint(vals)
        worst = max(worst, float(np.abs(vals - r).max()) if len(vals) else 0.0)
        out_data[lo:hi] = r.astype(np.int32)
    C = sp.csr_matrix((out_data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    C.eliminate_zeros()
    return C, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=str(INPUTS / "PancreasIntegratedAnnotated.h5ad"))
    ap.add_argument("--out", default=str(INPUTS / "PancreasCorrected.h5ad"))
    ap.add_argument("--table", default=str(HERE / "evidence" / "pancreas_sample_table.csv"))
    a = ap.parse_args()

    print(f"[build] reading {a.h5ad}", flush=True)
    A = ad.read_h5ad(a.h5ad)
    obs = A.obs
    tc = obs["total_counts"].to_numpy(np.float64)

    # ---- 1. raw counts ---------------------------------------------------
    C, worst = recover_counts(A.X, tc)
    row_sum = np.asarray(C.sum(axis=1)).ravel()
    nnz = np.diff(C.indptr)
    bad_sum = int((row_sum != np.rint(tc)).sum())
    bad_nnz = int((nnz != obs["n_genes_by_counts"].to_numpy()).sum())
    print(f"[build] counts recovered: worst rounding error {worst:.4f}; "
          f"rows with sum != total_counts: {bad_sum}; rows with nnz != n_genes_by_counts: {bad_nnz}", flush=True)
    if worst > 0.05 or bad_sum or bad_nnz:
        raise SystemExit("[build] raw counts are not exactly recoverable -- aborting")

    # ---- 2. identity of every row ---------------------------------------
    src = obs["SourceFile"].astype(str).str.replace("_matrix.mtx", "", regex=False).to_numpy()
    is_hpap = ~pd.Series(src).str.startswith("GSM").to_numpy()
    barcode = pd.Series(A.obs_names.astype(str)).str.extract(r"([ACGT]{14,16})", expand=False).to_numpy()
    donor = np.empty(len(src), dtype=object)
    library = np.empty(len(src), dtype=object)
    run = np.empty(len(src), dtype=object)
    for f in np.unique(src[is_hpap]):
        d, lib, r = parse_hpap(FILE_TO_HPAP[f])
        m = src == f
        donor[m], library[m], run[m] = d, lib, r
    donor[~is_hpap] = src[~is_hpap]          # one PDAC patient per GEO sample
    library[~is_hpap] = src[~is_hpap]
    run[~is_hpap] = src[~is_hpap]
    key = np.where(is_hpap, pd.Series(donor).astype(str) + "|" + pd.Series(barcode).astype(str),
                   pd.Series(A.obs_names.astype(str))).astype(str)
    if pd.isna(barcode[is_hpap]).any():
        raise SystemExit("[build] an HPAP cell has no parsable barcode")

    # ---- 3. merge runs of one library ------------------------------------
    codes, uniq = pd.factorize(key, sort=False)
    n_units = len(uniq)
    G = sp.csr_matrix((np.ones(len(codes), dtype=np.int32), (codes, np.arange(len(codes)))),
                      shape=(n_units, len(codes)))
    M = (G @ C).tocsr()
    M.sort_indices()
    print(f"[build] rows {A.n_obs} -> cells {n_units} "
          f"(HPAP rows {int(is_hpap.sum())} -> {len(np.unique(codes[is_hpap]))})", flush=True)

    ct = obs["celltype"].astype(str).to_numpy()
    lab = pd.DataFrame({"code": codes, "celltype": ct, "tc": tc})
    agg = (lab.groupby(["code", "celltype"], sort=False)
              .agg(n=("tc", "size"), tcmax=("tc", "max")).reset_index()
              .sort_values(["code", "n", "tcmax"], ascending=[True, False, False]))
    top = agg.drop_duplicates("code").set_index("code")
    n_copies = np.bincount(codes, minlength=n_units)
    agree = top["n"].reindex(range(n_units)).to_numpy() / n_copies

    first = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
    runs = pd.Series(run).groupby(codes).agg(lambda s: ";".join(sorted(set(map(str, s))))).to_numpy()
    new_obs = pd.DataFrame({
        "Sample": obs["Sample"].astype(str).to_numpy()[first],
        "group": obs["group"].astype(str).to_numpy()[first],
        "donor": donor[first].astype(str),
        "library": library[first].astype(str),
        "SourceFile": np.where(is_hpap[first], pd.Series(donor[first]).astype(str) + "_" +
                               pd.Series(library[first]).astype(str), src[first]),
        "runs": runs,
        "n_runs_merged": n_copies,
        "celltype": top["celltype"].reindex(range(n_units)).to_numpy(),
        "celltype_agreement": agree,
    })
    new_obs["total_counts"] = np.asarray(M.sum(axis=1)).ravel().astype(np.int64)
    new_obs["n_genes_by_counts"] = np.diff(M.indptr).astype(np.int64)
    names = np.where(is_hpap[first], pd.Series(barcode[first]).astype(str) + "-" + new_obs["donor"],
                     A.obs_names.astype(str).to_numpy()[first])
    new_obs.index = pd.Index(names, name=None)
    for c in ["Sample", "group", "donor", "library", "SourceFile", "celltype"]:
        new_obs[c] = new_obs[c].astype("category")

    # ---- 4. X = log1p(CP10k) from the merged counts ----------------------
    scale = TARGET_SUM / np.maximum(new_obs["total_counts"].to_numpy(np.float64), 1.0)
    Xn = M.astype(np.float64)
    Xn.data *= np.repeat(scale, np.diff(Xn.indptr))
    Xn.data = np.log1p(Xn.data)
    out = ad.AnnData(X=Xn.astype(np.float32), obs=new_obs, var=pd.DataFrame(index=A.var_names.copy()))
    out.layers["counts"] = M.astype(np.int32)
    out.uns["provenance"] = {
        "built_by": "scripts/build_pancreas_corrected.py", "date": str(date.today()),
        "source": Path(a.h5ad).name,
        "counts": "recovered exactly as round(expm1(X) * total_counts / 1e4)",
        "max_rounding_error": worst,
        "hpap_merge": "sequencing runs of one donor library summed by (donor, cell barcode)",
        "celltype": "majority label over merged copies; ties -> copy with most UMIs",
        "X": "log1p(normalize_total(counts, 1e4)), natural log",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(a.out, compression="lzf")
    print(f"[build] wrote {a.out}  {out.shape}", flush=True)

    # ---- 5. sample table (MO task 9) --------------------------------------
    rows = []
    for f in pd.unique(src):
        m = src == f
        hp = bool(is_hpap[m][0])
        rows.append({
            "source_file": f"{f}_matrix.mtx",
            "hpap_name": FILE_TO_HPAP.get(f, "") if hp else "",
            "condition": obs["Sample"].astype(str).to_numpy()[m][0],
            "series": "HPAP / PANC-DB" if hp else f.split("_")[0][:7],
            "donor": donor[m][0], "library": library[m][0], "run": run[m][0],
            "rows_in_original_object": int(m.sum()),
            "barcodes_also_in_other_runs": int(pd.Series(key[m]).isin(pd.Series(key[~m])).sum()) if hp else 0,
        })
    tab = pd.DataFrame(rows)
    per_donor = new_obs.groupby("donor", observed=True).size().rename("cells_after_merge")
    tab = tab.merge(per_donor, left_on="donor", right_index=True, how="left")
    tab = tab.sort_values(["condition", "donor", "run"]).reset_index(drop=True)
    tab.to_csv(a.table, index=False)
    print(f"[build] wrote {a.table}  ({len(tab)} files, {new_obs['donor'].nunique()} donors)")
    ref = new_obs[new_obs["Sample"].astype(str) == "Reference"]
    print(f"[build] Reference: {len(ref)} cells from {ref['donor'].nunique()} donors; "
          f"celltype agreement <1 in {int((ref['celltype_agreement'] < 1).sum())} merged cells")
    print(ref.groupby("donor", observed=True).size().to_string())


if __name__ == "__main__":
    main()
