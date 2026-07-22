#!/usr/bin/env python
"""
bench_mast_export.py
====================
Step 1 of the DUET vs R MAST head-to-head benchmark.

Exports ONE cell-type subset (optionally down-sampled to a cell budget) in a form
R/MAST can read, and runs DUET on the *identical* cells and genes, timing it.

Fairness rules, all deliberate:
  * SAME CELLS  - the subsample is drawn once, in Python, and both engines see it.
  * SAME GENES  - genes are filtered once here (DUET's mast_compat gate plus the
                  min.pct filter) and only the survivors are exported, so neither
                  engine is charged for filtering the other would not do.
  * SAME MODEL  - design is ~1+condition with no covariates, matching DUET's
                  mast_compat preset (which sets covariates=()). The R side must
                  use zlm(~condition) for this to mean anything.
  * TWO CLOCKS  - `duet_fit_s` is compute only; `duet_total_s` includes what DUET
                  does that MAST cannot avoid. The export written here is itself
                  the round-trip cost MAST pays and DUET does not, so it is timed
                  and reported separately as `export_s`.

Writes into <outdir>/<tag>/: expr.mtx, genes.txt, cells.txt, cond.txt, meta.json,
duet_results.csv
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite
import anndata as ad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_de_comparison as R          # noqa: E402  (reuse the harmonized runner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=str(HERE / "PancreasIntegratedAnnotated.h5ad"))
    ap.add_argument("--celltype", default="Ductal cell")
    ap.add_argument("--n-cells", type=int, default=0,
                    help="Down-sample to this many cells TOTAL (0 = all). Sampling is "
                         "stratified by condition so the group ratio is preserved.")
    ap.add_argument("--min-detect-frac", type=float, default=0.10)
    ap.add_argument("--outdir", default=str(HERE / "bench"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tag = a.tag or f"{a.celltype.replace(' ', '_')}_{a.n_cells or 'all'}"
    out = Path(a.outdir) / tag
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    print(f"[bench] loading {a.h5ad} (backed)", flush=True)
    A = ad.read_h5ad(a.h5ad, backed="r")
    ct_labels = A.obs["celltype"].astype(str).to_numpy()
    samp = A.obs["Sample"].astype(str)
    cond_all = np.where(samp.str.contains("Reference", case=False, na=False),
                        "Reference", "pancreas")

    mask = ct_labels == a.celltype
    idx = np.flatnonzero(mask)
    cond_ct = cond_all[mask]

    # stratified down-sample so the Reference:pancreas ratio is preserved
    if a.n_cells and a.n_cells < len(idx):
        keep = []
        for lab in ("Reference", "pancreas"):
            pool = np.flatnonzero(cond_ct == lab)
            take = int(round(a.n_cells * len(pool) / len(idx)))
            take = max(1, min(take, len(pool)))
            keep.append(rng.choice(pool, size=take, replace=False))
        sel = np.sort(np.concatenate(keep))
        idx = idx[sel]
        cond_ct = cond_ct[sel]
    print(f"[bench] {a.celltype}: {len(idx)} cells "
          f"(ref={int((cond_ct=='Reference').sum())}, test={int((cond_ct=='pancreas').sum())})",
          flush=True)

    view = A[idx]
    adm = ad.AnnData(X=view.X, obs=view.obs.copy(), var=A.var.copy())
    adm.obs["condition"] = cond_ct
    adm.obs["Sample"] = cond_ct
    n_ref = int((cond_ct == "Reference").sum())
    n_test = int((cond_ct == "pancreas").sum())

    # ---- run DUET on the full gene set, timed -------------------------------
    print("[bench] running DUET ...", flush=True)
    t0 = time.perf_counter()
    duet = R.run_duet(adm, a.celltype, sample_col="Sample", ref_label="Reference",
                      test_label="pancreas", n_ref=n_ref, n_test=n_test,
                      min_detect_frac=a.min_detect_frac)
    duet_fit_s = time.perf_counter() - t0
    genes = duet["gene"].astype(str).to_numpy()
    print(f"[bench] DUET: {len(genes)} genes tested in {duet_fit_s:.2f}s", flush=True)
    duet.to_csv(out / "duet_results.csv", index=False)

    # ---- export EXACTLY those genes for MAST --------------------------------
    print(f"[bench] exporting {len(genes)} genes x {len(idx)} cells for R ...", flush=True)
    t0 = time.perf_counter()
    gi = pd.Index(adm.var_names).get_indexer(genes)
    assert (gi >= 0).all(), "gene name mismatch between DUET output and var_names"
    X = adm.X
    X = X.tocsc() if sp.issparse(X) else sp.csc_matrix(X)
    Xg = X[:, gi].T.tocoo()          # genes x cells, as MAST expects
    mmwrite(str(out / "expr.mtx"), Xg, field="real", precision=7)
    np.savetxt(out / "genes.txt", genes, fmt="%s")
    np.savetxt(out / "cells.txt", np.asarray(adm.obs_names, dtype=str), fmt="%s")
    np.savetxt(out / "cond.txt", cond_ct, fmt="%s")
    export_s = time.perf_counter() - t0
    mtx_mb = (out / "expr.mtx").stat().st_size / 1e6
    print(f"[bench] export: {export_s:.2f}s, expr.mtx = {mtx_mb:.0f} MB", flush=True)

    meta = dict(celltype=a.celltype, tag=tag, n_cells=int(len(idx)),
                n_ref=n_ref, n_test=n_test, n_genes=int(len(genes)),
                duet_fit_s=round(duet_fit_s, 3), export_s=round(export_s, 3),
                mtx_mb=round(mtx_mb, 1), min_detect_frac=a.min_detect_frac,
                seed=a.seed, design="~1+condition (no covariates)")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"[bench] -> {out}")


if __name__ == "__main__":
    main()
