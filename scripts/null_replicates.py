#!/usr/bin/env python
"""
null_replicates.py — repeat the donor-swap null over many random splits.

The headline donor-swap numbers came from a single split per cell type. A single
split is one draw from a distribution with real variance (the same lesson the
pancreas Ductal permutation taught: 0.203 on one seed, 0.047 averaged over ten),
so the paper needs a mean and a spread rather than a point estimate.

For each seed the reference samples are re-partitioned at random and DUET,
Wilcoxon and DESeq2-pseudobulk are re-run on a matched gene set with BH-FDR
recomputed on the common genes.

Usage:
  python null_replicates.py --h5ad data/kang_8donors.h5ad --ref-label ctrl \
      --test-label stim --results-dir results/kang --tag kang --seeds 10
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ for sibling imports
import run_de_comparison as R          # noqa: E402

warnings.filterwarnings("ignore")
ALPHA = 0.05


def _bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n, float)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--ref-label", required=True)
    ap.add_argument("--test-label", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--celltype-col", default="celltype")
    ap.add_argument("--sample-col", default="Sample")
    ap.add_argument("--source-col", default="SourceFile")
    ap.add_argument("--outdir", default=str(HERE / "results" / "null"))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    duet = pd.read_csv(Path(a.results_dir) / "de_DUET_all_celltypes.csv", low_memory=False)
    celltypes = list(duet[duet.tested == True]["celltype"].dropna().unique())   # noqa: E712

    A = ad.read_h5ad(a.h5ad, backed="r")
    samp = A.obs[a.sample_col].astype(str)
    real = np.where(samp.str.contains(a.ref_label, case=False, na=False),
                    a.ref_label, a.test_label)
    ctl = A.obs[a.celltype_col].astype(str).to_numpy()
    src = A.obs[a.source_col].astype(str).to_numpy()

    rows = []
    for ct in celltypes:
        mask = ctl == ct
        view = A[mask]
        base = ad.AnnData(X=view.X, obs=view.obs.copy(), var=A.var.copy())
        if "counts" in A.layers:
            base.layers["counts"] = view.layers["counts"]
        ref_cells = real[mask] == a.ref_label
        donors = np.unique(src[mask][ref_cells])
        print(f"\n=== [{a.tag}] {ct}: {int(mask.sum())} cells, "
              f"{len(donors)} {a.ref_label} donors ===", flush=True)

        for seed in range(a.seeds):
            rng = np.random.default_rng(1000 + seed)
            d = donors.copy(); rng.shuffle(d)
            half = len(d) // 2
            gA = set(d[:half])
            swap = np.where(np.isin(src[mask], list(gA)), a.ref_label, a.test_label)
            swap[~ref_cells] = ""
            keep = swap != ""
            sub = base[keep].copy()
            c = swap[keep]
            sub.obs["condition"] = c
            sub.obs[a.sample_col] = c
            sub.obs[a.source_col] = src[mask][keep]
            nr = int((c == a.ref_label).sum()); nt = int((c == a.test_label).sum())
            if nr < 50 or nt < 50:
                print(f"  seed {seed}: skipped (nr={nr}, nt={nt})", flush=True)
                continue

            per = {}
            try:
                per["DUET"] = R.run_duet(sub, ct, sample_col=a.sample_col,
                                         ref_label=a.ref_label, test_label=a.test_label,
                                         n_ref=nr, n_test=nt)
                per["wilcoxon"] = R.run_wilcoxon(sub, ct, ref_label=a.ref_label,
                                                 test_label=a.test_label,
                                                 n_ref=nr, n_test=nt)
                if "counts" in sub.layers:
                    per["DESeq2-pseudobulk"] = R.run_deseq2(
                        sub, ct, counts_layer="counts", sample_col=a.source_col,
                        condition=c, ref_label=a.ref_label, test_label=a.test_label,
                        n_ref=nr, n_test=nt, min_samples_per_group=2,
                        min_gene_counts=10, min_cells_per_sample=10)
            except Exception as e:
                print(f"  seed {seed}: FAILED {type(e).__name__}: {e}", flush=True)
                continue

            per = {k: v[v.tested == True].dropna(subset=["pvalue"])       # noqa: E712
                   for k, v in per.items()}
            per = {k: v for k, v in per.items() if len(v)}
            if len(per) < 2:
                continue
            common = set.intersection(*(set(v.gene) for v in per.values()))
            line = []
            for m, v in per.items():
                v = v[v.gene.isin(common)].drop_duplicates("gene")
                p = v["pvalue"].to_numpy(float)
                fdr = _bh(p)
                rows.append(dict(dataset=a.tag, celltype=ct, seed=seed, method=m,
                                 n_common=len(common),
                                 pct_sig_fdr05=round(100 * float((fdr < 0.05).mean()), 3),
                                 type1_at_alpha05=round(float((p < ALPHA).mean()), 4)))
                line.append(f"{m}={rows[-1]['pct_sig_fdr05']:.1f}%")
            print(f"  seed {seed} ({nr}v{nt}): " + "  ".join(line), flush=True)
            del sub
        del base

    D = pd.DataFrame(rows)
    D.to_csv(out / f"null_reps_{a.tag}.csv", index=False)
    print("\n" + "=" * 78)
    print(f"DONOR-SWAP NULL OVER {a.seeds} RANDOM SPLITS — {a.tag}")
    print("=" * 78)
    s = (D.groupby(["celltype", "method"])["pct_sig_fdr05"]
           .agg(["count", "mean", "std", "min", "max"]).round(2))
    print(s.to_string())
    print(f"\n-> {out / f'null_reps_{a.tag}.csv'}")


if __name__ == "__main__":
    main()
