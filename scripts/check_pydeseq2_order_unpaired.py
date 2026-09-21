#!/usr/bin/env python
"""
check_pydeseq2_order_unpaired.py -- are the unpaired (~condition) PyDESeq2 fits
behind the donor partitions and Table S9 sensitive to sample order?

Companion to check_pydeseq2_order.py (paired Kang design). Here every design is
~condition, as in the negative-control donor partitions (Table 6) and the Crowell
and pancreas rows of Table S9. For each fit the pseudobulk matrix is built with
duet.aggregate_pseudobulk (explicit factor levels, counts and metadata aligned)
and fitted with PyDESeq2 twice: in the canonical sorted order the wrapper uses, and
reversed. Rejections are BH-adjusted over the genes PyDESeq2 tested (the
partition script adjusts over the genes shared with DUET and Wilcoxon, so counts
here are comparable between orders, not to Table 6 itself).

Output: evidence/pydeseq2_order/unpaired_order.csv
Usage:  python scripts/check_pydeseq2_order_unpaired.py [--datasets kang,crowell,pancreas]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.background(1)

import argparse                                            # noqa: E402
import warnings                                            # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from donor_swap_partitions import DATASETS, MIN_CELLS_PER_DONOR, unique_splits, bh   # noqa: E402

warnings.filterwarnings("ignore")
ALPHA = 0.05
TEST = {"kang": "stim", "crowell": "LPS", "pancreas": "pancreas"}


def fit(pb, meta, ref, test):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    assert list(pb.index) == list(meta.index)
    dds = DeseqDataSet(counts=pb, metadata=meta, design="~condition", quiet=True, n_cpus=1)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", test, ref], quiet=True, n_cpus=1)
    st.summary()
    return st.results_df


def compare(A, donor_col, cond_col, ref, test, **info):
    from duet import aggregate_pseudobulk
    pb, meta = aggregate_pseudobulk(A, donor_col=donor_col, condition_col=cond_col, ref_label=ref,
                                    test_label=test, min_cells_per_donor=MIN_CELLS_PER_DONOR)
    pb = pb.loc[:, pb.sum(axis=0) >= 10]
    a = fit(pb, meta, ref, test)
    rev = pb.index[::-1]
    b = fit(pb.loc[rev], meta.loc[rev], ref, test).loc[a.index]
    ok = a.pvalue.notna() & b.pvalue.notna()
    qa, qb = bh(a.pvalue[ok]), bh(b.pvalue[ok])
    ra, rb = qa < ALPHA, qb < ALPHA
    return dict(**info, n_samples=len(pb), n_genes=int(ok.sum()),
                max_abs_dlog2FC=float((a.log2FoldChange - b.log2FoldChange).abs().max()),
                max_abs_dpvalue=float((a.pvalue - b.pvalue).abs().max()),
                rejections_sorted=int(ra.sum()), rejections_reversed=int(rb.sum()),
                genes_changing_call=int((ra != rb).sum()),
                any_rejection_sorted=bool(ra.any()), any_rejection_reversed=bool(rb.any()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="kang,crowell,pancreas")
    ap.add_argument("--out", default=str(HERE / "evidence" / "pydeseq2_order" / "unpaired_order.csv"))
    a = ap.parse_args()
    rows = []
    for name in a.datasets.split(","):
        cfg = DATASETS[name]
        for ct in cfg["celltypes"]:
            A = _runtime.load_rows(cfg["h5ad"], lambda o: (o["celltype"].astype(str) == ct).to_numpy())
            A.obs[cfg["cond"]] = A.obs[cfg["cond"]].astype(str)
            if name != "kang":           # Table S9: tumor/LPS vs reference, one condition per donor
                rows.append(compare(A, cfg["donor"], cfg["cond"], cfg["ref"], TEST[name],
                                    analysis="Table S9 main contrast", dataset=name, celltype=ct, split=0))
                print(rows[-1], flush=True)
            sub = A[(A.obs[cfg["cond"]] == cfg["ref"]).to_numpy()].copy()
            counts = sub.obs[cfg["donor"]].astype(str).value_counts()
            donors = sorted(counts.index[counts >= MIN_CELLS_PER_DONOR])
            for i, (arm_a, arm_b) in enumerate(unique_splits(donors)):
                d = sub.obs[cfg["donor"]].astype(str)
                S = sub[d.isin(arm_a + arm_b).to_numpy()].copy()
                S.obs["condition"] = np.where(S.obs[cfg["donor"]].astype(str).isin(arm_a), "armA", "armB")
                rows.append(compare(S, cfg["donor"], "condition", "armA", "armB",
                                    analysis="donor partition", dataset=name, celltype=ct, split=i + 1))
            part = pd.DataFrame([r for r in rows if r["analysis"] == "donor partition"
                                 and r["dataset"] == name and r["celltype"] == ct])
            print(f"[unpaired] {name}/{ct}: {len(part)} partitions, "
                  f"{int(part.genes_changing_call.sum())} call changes, "
                  f"{int((part.any_rejection_sorted != part.any_rejection_reversed).sum())} any-rejection changes",
                  flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[unpaired] wrote {a.out}")


if __name__ == "__main__":
    main()
