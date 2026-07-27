#!/usr/bin/env python
"""
validate_pseudobulk_null.py — does the combined caller actually control the null?

The claim the pseudobulk arm exists to make good is narrow and testable: on two
groups that differ by nothing, the cell-level hurdle calls a large and erratic
fraction of genes differential, and the sample-level arm does not. This runs
that experiment through `run_duet_pseudobulk` itself rather than through the
benchmark harness, so what is measured is the shipped code path.

Reference samples only, split at random into two fake arms over several seeds.
There is no condition effect anywhere, so every call is a false positive and the
correct answer is ~0%.

Usage: python scripts/validate_pseudobulk_null.py [--seeds 5]
"""
from __future__ import annotations
import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(HERE))
from duet import run_duet_pseudobulk                        # noqa: E402

DATASETS = {
    "kang": dict(h5ad=INPUTS / "data" / "kang_8donors.h5ad",
                 donor="replicate", cond="Sample", ref="ctrl",
                 celltypes=["CD14+ Monocytes", "B cells"]),
    "crowell": dict(h5ad=INPUTS / "data" / "crowell_4vs4.h5ad",
                    donor="SourceFile", cond="Sample", ref="Vehicle",
                    celltypes=["Astrocytes", "Excit. Neuron"]),
}


def one(A, donor_col, celltype, seed):
    """Split this cell type's donors at random into two arms that differ by
    nothing, then run both DUET arms on the result."""
    rng = np.random.default_rng(seed)
    donors = pd.unique(A.obs[donor_col].astype(str))
    if len(donors) < 4:
        return None
    shuffled = rng.permutation(donors)
    half = len(shuffled) // 2
    grp = {d: ("fakeA" if i < half else "fakeB")
           for i, d in enumerate(shuffled)}
    B = A.copy()
    B.obs["fake_condition"] = B.obs[donor_col].astype(str).map(grp)
    B = B[B.obs.fake_condition.notna()].copy()

    with tempfile.TemporaryDirectory() as tmp:
        out = run_duet_pseudobulk(
            B, output_dir=tmp, celltype_col="celltype",
            condition_col="fake_condition", ref_label="fakeA", test_label="fakeB",
            donor_col=donor_col, counts_layer="counts",
            mast_compat=True, output_name="x.csv")
        d = pd.read_csv(out)

    t = d[d.tested == True]                                  # noqa: E712
    pbt = d[d.pb_tested == True]                             # noqa: E712
    if not len(t) or not len(pbt):
        return None
    return dict(
        celltype=celltype, seed=seed,
        n_donors_ref=int(pbt.n_donors_ref.iloc[0]),
        n_donors_test=int(pbt.n_donors_test.iloc[0]),
        duet_pct=round(100 * float((t.fdr < 0.05).mean()), 3),
        pseudobulk_pct=round(100 * float((pbt.pb_padj < 0.05).mean()), 3),
        n_significant=int((d.call == "significant").sum()),
        n_ranked_only=int((d.call == "ranked_only").sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--datasets", default="kang,crowell")
    ap.add_argument("--out", default=str(HERE / "evidence" / "pseudobulk_null.csv"))
    a = ap.parse_args()

    rows = []
    for name in [s.strip() for s in a.datasets.split(",")]:
        cfg = DATASETS[name]
        if not Path(cfg["h5ad"]).is_file():
            print(f"[{name}] {cfg['h5ad']} missing, skipped")
            continue
        A0 = sc.read_h5ad(cfg["h5ad"])
        # reference arm only: no real condition effect can survive this
        A0 = A0[A0.obs[cfg["cond"]].astype(str) == cfg["ref"]].copy()
        for ct in cfg["celltypes"]:
            sub = A0[A0.obs.celltype.astype(str) == ct].copy()
            print(f"\n[{name}/{ct}] {sub.n_obs} cells, "
                  f"{sub.obs[cfg['donor']].nunique()} donors", flush=True)
            for seed in range(a.seeds):
                r = one(sub, cfg["donor"], ct, seed)
                if r:
                    r["dataset"] = name
                    rows.append(r)
                    print(f"   seed {seed}: DUET {r['duet_pct']:6.2f}%   "
                          f"pseudobulk {r['pseudobulk_pct']:5.2f}%", flush=True)

    if not rows:
        raise SystemExit("no results")
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    print("\n" + "=" * 62)
    print("DONOR-SWAP NULL — the correct answer is ~0% for both columns")
    print("=" * 62)
    g = df.groupby("dataset")[["duet_pct", "pseudobulk_pct"]].agg(["mean", "max"])
    print(g.round(3).to_string())
    print(f"\noverall   DUET mean {df.duet_pct.mean():.2f}%  max {df.duet_pct.max():.2f}%")
    print(f"          pseudobulk mean {df.pseudobulk_pct.mean():.3f}%  "
          f"max {df.pseudobulk_pct.max():.3f}%")
    print(f"          splits above 5%:  DUET {(df.duet_pct > 5).sum()}/{len(df)}, "
          f"pseudobulk {(df.pseudobulk_pct > 5).sum()}/{len(df)}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
