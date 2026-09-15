"""
arm_offset.py -- per replicate, on the TRUE-NULL genes (ee and ep, decision A4): how
far apart are the two arms, (i) at the donor level in units of the donor-level
standard error -- what a sample-level test discounts -- and (ii) at the cell level as
a standardised difference, which a cell-level test multiplies by sqrt(N)?

If results/fdr_scale/fdr_scale_diag_summary.csv exists, also reports the Spearman
correlation between each replicate's realised cell-level offset and each method's
raw type-I error on the true nulls (the correlations quoted with Table 6).

Output: results/fdr_scale/arm_offset.csv (+ arm_offset_vs_typeI.csv)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.background(1)

import os                                                  # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import scipy.sparse as sp                                  # noqa: E402
from scipy.stats import spearmanr                          # noqa: E402

REPO = Path(__file__).resolve().parent.parent
INPUTS = Path(os.environ.get("DUET_INPUTS", REPO.parent / "inputs"))
from evaluate_sim_replicates import build                  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "fdr_scale"
out.mkdir(parents=True, exist_ok=True)
rows = []
for rep in sorted(INPUTS.glob("data/sim_rep*")):
    if not (rep / "seed.txt").is_file():
        continue
    A = build(rep)
    null = A.var.index[A.var.category.isin(["ee", "ep"])]
    gi = pd.Index(A.var_names).get_indexer(null.astype(str))
    X = (A.X.tocsc() if sp.issparse(A.X) else sp.csc_matrix(A.X))[:, gi]
    smp, grp = A.obs["SourceFile"].astype(str).to_numpy(), A.obs["Sample"].astype(str).to_numpy()
    M, G = [], []
    for s in pd.unique(smp):
        m = smp == s
        M.append(np.asarray(X[m].mean(axis=0)).ravel())
        G.append(grp[m][0])
    M, G = np.vstack(M), np.asarray(G)
    a, b = M[G == "A"], M[G == "B"]
    sp2 = ((a.var(0, ddof=1) * (len(a) - 1)) + (b.var(0, ddof=1) * (len(b) - 1))) / (len(a) + len(b) - 2)
    tstat = (b.mean(0) - a.mean(0)) / np.sqrt(np.maximum(sp2, 1e-12) * (1 / len(a) + 1 / len(b)))  # ~t_6
    ca = np.asarray(X[grp == "A"].mean(axis=0)).ravel()
    cb = np.asarray(X[grp == "B"].mean(axis=0)).ravel()
    sd = np.sqrt(np.asarray(X.multiply(X).mean(axis=0)).ravel() - np.asarray(X.mean(axis=0)).ravel() ** 2)
    d_cell = (cb - ca) / np.maximum(sd, 1e-9)
    rows.append(dict(replicate=rep.name, n_null=len(null), mean_abs_t_donor=float(np.abs(tstat).mean()),
                     frac_t_gt2=float((np.abs(tstat) > 2).mean()), mean_abs_d_cell=float(np.abs(d_cell).mean()),
                     median_abs_d_cell=float(np.median(np.abs(d_cell))),
                     frac_d_gt_003=float((np.abs(d_cell) > 0.03).mean())))
    print(rows[-1], flush=True)
    del A
D = pd.DataFrame(rows)
D.to_csv(out / "arm_offset.csv", index=False)
diag = out / "fdr_scale_diag_summary.csv"
if diag.is_file():
    S = pd.read_csv(diag).merge(D, on="replicate")
    cor = []
    for m, g in S.groupby("method"):
        r = spearmanr(g["median_abs_d_cell"], g["typeI_raw_p05"])
        cor.append(dict(method=m, n_replicates=len(g), spearman_rho=r.statistic, p_value=r.pvalue))
    pd.DataFrame(cor).to_csv(out / "arm_offset_vs_typeI.csv", index=False)
    print(pd.DataFrame(cor).round(4).to_string(index=False))
