#!/usr/bin/env python
"""
null_calibration.py
===================
Type-I error calibration for DUET, on any dataset, under two nulls with NO real
condition effect. The correct answer in both is ~0 significant genes.

  (1) PERMUTATION NULL  - shuffle the condition label across cells within a cell
      type. Destroys the condition effect but ALSO destroys the donor structure,
      so it is the easy null: it only detects a broken test.

  (2) DONOR-SWAP NULL   - take the reference condition ONLY and split its donors
      into two pseudo-groups at the sample level. Real donor-to-donor variation
      survives, which is exactly the variance component cell-level methods ignore
      (Squair et al. 2021, Nat Commun 12:5692; Zimmerman et al. 2021, 12:738).
      A method that passes (1) but fails (2) is pseudoreplicating.

Methods are compared on a MATCHED gene set with BH-FDR recomputed on the common
genes: DUET applies a min.pct filter and tests a few thousand genes while scanpy's
Wilcoxon tests all of them (mostly all-zero, returning p=1), and comparing raw
rates across those denominators badly flatters Wilcoxon.

Because the donor-swap null is a property of the experimental DESIGN, not of the
method, running it across datasets with different designs is the point:
  pancreas - across-donor, condition fully confounded with donor
  Kang     - paired within-donor (every donor in both arms)
  Crowell  - across-sample but balanced 4v4

Usage:
  python null_calibration.py --h5ad data/kang_8donors.h5ad \
      --ref-label ctrl --test-label stim --tag kang
"""
from __future__ import annotations
import argparse, sys, time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import kstest

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ for sibling imports
import run_de_comparison as R          # noqa: E402

ALPHA = 0.05


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n, float)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def summarize(df, method, celltype, design, tag):
    d = df[df["tested"] == True].dropna(subset=["pvalue"])      # noqa: E712
    p = d["pvalue"].to_numpy(float)
    p = p[np.isfinite(p)]
    if not len(p):
        return dict(dataset=tag, design=design, method=method, celltype=celltype, n_tested=0)
    fdr = pd.to_numeric(d["fdr"], errors="coerce").to_numpy(float)
    return dict(
        dataset=tag, design=design, method=method, celltype=celltype, n_tested=len(p),
        n_sig_fdr05=int(np.nansum(fdr < 0.05)),
        pct_sig_fdr05=round(100 * float(np.nansum(fdr < 0.05)) / len(p), 3),
        type1_at_alpha05=round(float((p < ALPHA).mean()), 4),
        ks_stat=round(float(kstest(p, "uniform").statistic), 4),
        median_p=round(float(np.median(p)), 4),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=str(INPUTS / "PancreasIntegratedAnnotated.h5ad"))
    ap.add_argument("--ref-label", default="Reference")
    ap.add_argument("--test-label", default="pancreas")
    ap.add_argument("--celltype-col", default="celltype")
    ap.add_argument("--sample-col", default="Sample")
    ap.add_argument("--source-col", default="SourceFile")
    ap.add_argument("--celltypes", default=None, help="comma-separated; default = "
                    "the cell types in --results-dir/de_DUET_all_celltypes.csv")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--tag", default="dataset")
    ap.add_argument("--outdir", default=str(HERE / "results" / "null"))
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    if a.celltypes:
        celltypes = [c.strip() for c in a.celltypes.split(",") if c.strip()]
    elif a.results_dir:
        d = pd.read_csv(Path(a.results_dir) / "de_DUET_all_celltypes.csv", low_memory=False)
        celltypes = list(d[d.tested == True]["celltype"].dropna().unique())   # noqa: E712
    else:
        raise SystemExit("give --celltypes or --results-dir")
    print(f"[null] {a.tag}: cell types {celltypes}")

    A = ad.read_h5ad(a.h5ad, backed="r")
    obs = A.obs
    samp = obs[a.sample_col].astype(str)
    real_cond = np.where(samp.str.contains(a.ref_label, case=False, na=False),
                         a.ref_label, a.test_label)
    ct_labels = obs[a.celltype_col].astype(str).to_numpy()
    src = obs[a.source_col].astype(str).to_numpy()

    rows, per_gene = [], []
    for ct in celltypes:
        mask = ct_labels == ct
        print(f"\n=== [{a.tag}] {ct}: {int(mask.sum())} cells ===", flush=True)
        view = A[mask]
        adm = ad.AnnData(X=view.X, obs=view.obs.copy(), var=A.var.copy())
        if "counts" in A.layers:                 # needed for the pseudobulk arm
            adm.layers["counts"] = view.layers["counts"]

        perm = rng.permutation(real_cond[mask])

        ref_cells = real_cond[mask] == a.ref_label
        ref_donors = np.unique(src[mask][ref_cells])
        rng.shuffle(ref_donors)
        half = len(ref_donors) // 2
        gA = set(ref_donors[:half])
        swap = np.where(np.isin(src[mask], list(gA)), a.ref_label, a.test_label)
        swap[~ref_cells] = ""
        print(f"  donor-swap: {len(ref_donors)} {a.ref_label} donors -> "
              f"{half} vs {len(ref_donors)-half}", flush=True)

        for design, cond in (("permutation", perm), ("donor_swap", swap)):
            keep = cond != ""
            sub = adm[keep].copy()
            c = cond[keep]
            sub.obs["condition"] = c
            sub.obs[a.sample_col] = c
            nr = int((c == a.ref_label).sum()); nt = int((c == a.test_label).sum())
            if nr < a.min_cells or nt < a.min_cells:
                print(f"  [{design}] skipped (nr={nr}, nt={nt})", flush=True)
                continue
            print(f"  [{design}] ref={nr} test={nt}", flush=True)

            # Pseudobulk arm: the real donor id is the aggregation unit. Because
            # run_deseq2 groups by (sample x condition), both nulls work naturally.
            # Donor-swap: each donor sits wholly in one pseudo-group -> a normal
            # unpaired 2-group pseudobulk. Permutation: each donor is split across
            # both pseudo-groups -> two pseudobulk samples per donor, i.e. a paired
            # null. This arm is the control: it should be the one that PASSES.
            sub.obs[a.source_col] = src[mask][keep]
            methods = [
                ("DUET", lambda: R.run_duet(sub, ct, sample_col=a.sample_col,
                                            ref_label=a.ref_label, test_label=a.test_label,
                                            n_ref=nr, n_test=nt)),
                ("wilcoxon", lambda: R.run_wilcoxon(sub, ct, ref_label=a.ref_label,
                                                    test_label=a.test_label,
                                                    n_ref=nr, n_test=nt)),
            ]
            if "counts" in sub.layers:
                methods.append(
                    ("DESeq2-pseudobulk", lambda: R.run_deseq2(
                        sub, ct, counts_layer="counts", sample_col=a.source_col,
                        condition=c, ref_label=a.ref_label, test_label=a.test_label,
                        n_ref=nr, n_test=nt, min_samples_per_group=2,
                        min_gene_counts=10, min_cells_per_sample=10)))
            for mname, fn in methods:
                t0 = time.time()
                try:
                    r = fn()
                    per_gene.append(r.assign(_design=design, _method=mname))
                    s = summarize(r, mname, ct, design, a.tag)
                    s["seconds"] = round(time.time() - t0, 1)
                    rows.append(s)
                    print(f"    {mname:9s} n_sig={s.get('n_sig_fdr05','?'):>6} "
                          f"({s.get('pct_sig_fdr05','?')}%)  type1={s.get('type1_at_alpha05')} "
                          f"medP={s.get('median_p')}  {s['seconds']}s", flush=True)
                except Exception as e:
                    print(f"    {mname}: FAILED {type(e).__name__}: {e}", flush=True)
            del sub
        del adm

    pd.DataFrame(rows).to_csv(out / f"null_{a.tag}.csv", index=False)

    # ---- matched-gene comparison ----
    pg = pd.concat(per_gene, ignore_index=True)
    pg = pg[pg["tested"] == True].dropna(subset=["pvalue"])       # noqa: E712
    mrows = []
    for (design, ct), g in pg.groupby(["_design", "celltype"]):
        # intersect over EVERY method present, so the pseudobulk arm is included
        by_method = {m: sub for m, sub in g.groupby("_method") if len(sub)}
        if len(by_method) < 2:
            continue
        common = set.intersection(*(set(sub.gene) for sub in by_method.values()))
        if not common:
            continue
        for mname, d in by_method.items():
            d = d[d.gene.isin(common)]
            p = d["pvalue"].to_numpy(float)
            fdr = _bh(p)
            mrows.append(dict(dataset=a.tag, design=design, celltype=ct, method=mname,
                              n_common=len(common),
                              n_sig_fdr05=int(np.nansum(fdr < 0.05)),
                              pct_sig_fdr05=round(100*float(np.nansum(fdr < 0.05))/len(p), 2),
                              type1_at_alpha05=round(float((p < ALPHA).mean()), 4),
                              median_p=round(float(np.median(p)), 4)))
    M = pd.DataFrame(mrows)
    M.to_csv(out / f"null_{a.tag}_matched.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + "=" * 78)
    print(f"MATCHED-GENE NULL CALIBRATION - {a.tag}")
    print("  n_sig should be ~0 | type1 ~0.05 | median_p ~0.5")
    print("=" * 78)
    print(M.to_string(index=False))
    print(f"\n-> {out / f'null_{a.tag}_matched.csv'}")


if __name__ == "__main__":
    main()
