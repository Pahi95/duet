#!/usr/bin/env python
"""
donor_swap_partitions.py -- the donor-swap global null over every
unique balanced donor partition, with the full DUET decision rule.

Only reference-arm cells are used. Every cell stays with its donor and all of a
donor's cells go to the same arm, so there is no condition effect anywhere and every
call is a false positive. Partitions: all unique splits into two arms of k donors
(k = n // 2); with an odd number of donors each split leaves one donor out. A split
and its mirror image are one split. Per cell type: Kang 8 donors -> 35 splits,
Crowell 4 animals -> 3, pancreas 5 donors -> 15 (decision A2b).

Per split: DUET (mast_compat), Wilcoxon (scanpy), DESeq2 pseudobulk
(duet.run_pseudobulk: raw counts summed per donor), and the combined rule (decision
A3): significant = pseudobulk padj < 0.05 and the same sign as the DUET coefficient.
BH is recomputed on the genes all three methods tested, so the percentages share one
denominator. This replaces the earlier 90-split (null_replicates.py) and 20-split
(validate_pseudobulk_null.py) runs, which drew splits at random and could repeat one.

Reported per split: the number of BH discoveries and the rejection proportion
(discoveries / tested genes). Neither is a false discovery rate -- under a global null
any discovery is false. The family-wise summary is the share of splits with at least
one discovery.

Outputs: results/null/donor_swap/<dataset>__<celltype>.csv (one row per split x method)
         results/null/donor_swap_summary.csv
Usage:   python scripts/donor_swap_partitions.py [--datasets kang,crowell,pancreas] [--workers 3]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.limit_threads(1)

import argparse                                            # noqa: E402
import itertools                                           # noqa: E402
import os                                                  # noqa: E402
import tempfile                                            # noqa: E402
import time                                                # noqa: E402
import warnings                                            # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed   # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent              # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(HERE))
ALPHA = 0.05
MIN_CELLS_PER_DONOR = 10

DATASETS = {
    "kang": dict(h5ad=INPUTS / "data" / "kang_8donors.h5ad", donor="replicate", cond="Sample", ref="ctrl",
                 celltypes=["B cells", "CD14+ Monocytes", "CD4 T cells"]),
    "crowell": dict(h5ad=INPUTS / "data" / "crowell_4vs4.h5ad", donor="SourceFile", cond="Sample", ref="Vehicle",
                    celltypes=["Astrocytes", "Excit. Neuron", "Inhib. Neuron"]),
    "pancreas": dict(h5ad=INPUTS / "PancreasCorrected.h5ad", donor="donor", cond="Sample", ref="Reference",
                     celltypes=["Ductal cell", "Endothelial cell", "Stellate cell"]),
}


def unique_splits(donors: list[str]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """All unique balanced two-arm splits; odd n leaves one donor out per split."""
    donors = sorted(donors)
    pools = [donors] if len(donors) % 2 == 0 else [[d for d in donors if d != out] for out in donors]
    splits = []
    for pool in pools:
        k = len(pool) // 2
        first = pool[0]                   # fixing one donor in arm A removes mirror duplicates
        for arm_a in itertools.combinations(pool, k):
            if first not in arm_a:
                continue
            arm_b = tuple(d for d in pool if d not in arm_a)
            splits.append((arm_a, arm_b))
    return splits


def bh(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    n = int(ok.sum())
    if n:
        o = np.argsort(p[ok])
        r = (p[ok][o] * n / np.arange(1, n + 1))[::-1]
        qq = np.empty(n)
        qq[o] = np.minimum(np.minimum.accumulate(r)[::-1], 1.0)
        q[ok] = qq
    return q


def run_split(sub, donor_col, arm_a, arm_b) -> dict[str, pd.DataFrame]:
    from duet import run_duet
    from duet.pseudobulk import run_pseudobulk
    import run_de_comparison as R
    d = sub.obs[donor_col].astype(str)
    keep = d.isin(arm_a + arm_b).to_numpy()
    A = sub[keep].copy()
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
    q = {"DUET": bh(du.loc[common, "pvalue"]), "wilcoxon": bh(wi.loc[common, "pvalue"]),
         "DESeq2-pseudobulk": bh(pb.loc[common, "pb_pvalue"])}
    same_dir = np.sign(pb.loc[common, "pb_log2FC"].to_numpy()) == np.sign(du.loc[common, "coef"].to_numpy())
    q["DUET+PB"] = np.where((q["DESeq2-pseudobulk"] < ALPHA) & same_dir, 0.0, 1.0)
    return dict(q=q, n_common=len(common), n_a=n_a, n_b=n_b,
                n_direction_conflict=int(((q["DESeq2-pseudobulk"] < ALPHA) & ~same_dir).sum()))


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
    counts = sub.obs[cfg["donor"]].astype(str).value_counts()
    donors = sorted(counts.index[counts >= MIN_CELLS_PER_DONOR])
    rows = []
    for i, (arm_a, arm_b) in enumerate(unique_splits(donors)):
        r = run_split(sub, cfg["donor"], arm_a, arm_b)
        for m, q in r["q"].items():
            n_disc = int(np.nansum(q < ALPHA))
            rows.append(dict(dataset=name, celltype=celltype, split=i + 1, arm_a=";".join(arm_a),
                             arm_b=";".join(arm_b), n_cells_a=r["n_a"], n_cells_b=r["n_b"], method=m,
                             n_tested=r["n_common"], n_discoveries=n_disc,
                             rejection_pct=100 * n_disc / max(r["n_common"], 1), any_discovery=n_disc > 0,
                             n_direction_conflict=r["n_direction_conflict"] if m == "DUET+PB" else np.nan))
    df = pd.DataFrame(rows)
    tmp = dest.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    os.replace(tmp, dest)
    return (f"[{name}/{celltype}] {len(donors)} donors, {df.split.nunique()} splits, "
            f"{sub.n_obs} cells, {time.time() - t0:.0f}s")


def summarise(outdir: Path) -> pd.DataFrame:
    D = pd.concat([pd.read_csv(f) for f in sorted(outdir.glob("*.csv"))], ignore_index=True)
    g = D.groupby(["dataset", "celltype", "method"], sort=False)
    S = g.agg(n_splits=("split", "nunique"), n_tested_median=("n_tested", "median"),
              rejection_pct_mean=("rejection_pct", "mean"), rejection_pct_sd=("rejection_pct", "std"),
              rejection_pct_median=("rejection_pct", "median"), rejection_pct_min=("rejection_pct", "min"),
              rejection_pct_max=("rejection_pct", "max"), discoveries_median=("n_discoveries", "median"),
              discoveries_min=("n_discoveries", "min"), discoveries_max=("n_discoveries", "max"),
              share_splits_with_any_discovery=("any_discovery", "mean")).reset_index()
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="kang,crowell,pancreas")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default=str(HERE / "results" / "null" / "donor_swap"))
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
                print(f"[donor-swap] FAIL {type(e).__name__}: {e}", flush=True)
    S = summarise(out)
    S.to_csv(out.parent / "donor_swap_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S[["dataset", "celltype", "method", "n_splits", "rejection_pct_mean", "rejection_pct_median",
             "rejection_pct_max", "share_splits_with_any_discovery"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
