#!/usr/bin/env python
"""
make_fig3_effects.py -- Figure 3, rebuilt as MO task 15 asks (decision A5).

Values computed from ONE input, the corrected pancreas object
(X = log1p(normalize_total(counts, 1e4)), natural log):
  common log2FC    log2[(expm1(xbar_test) + 1e-9) / (expm1(xbar_ref) + 1e-9)], xbar = mean of
                   log1p(CP10k) over the group's cells (scanpy's definition)
  delta detection  100 * (p_test - p_ref), p = share of the group's cells with count > 0,
                   in percentage points
Method outputs, each on its own panel, colour scale and unit (never a shared scale):
  DUET      coef / ln 2   difference of the per-group log1p means (MAST-style, compressed)
  MAST      logFC / ln 2  the same quantity from R MAST 1.33.0
  diffxpy   log2FC        condition coefficient of a cell-level NB GLM (Wald)
  DESeq2    log2FC        pseudobulk NB GLM, one profile per donor, unshrunken MLE
  Wilcoxon  log2FC        scanpy rank_genes_groups (the same formula as the common log2FC)
Gene selection, fixed before plotting: genes detected in >= 10% of the cells of at
least one group in each of the three cell types and tested by all five methods in all
three; the 20 highest and 20 lowest by the mean common log2FC over the cell types.
No significance filter, so no method decides which genes are shown.

The tumour-vs-non-tumour contrast is perfectly confounded with data source (all 45
tumour samples from five GEO series, all non-tumour cells from HPAP), which the
condition-by-source table written here documents; the figure is descriptive.

Outputs: evidence/figures/fig3_effects.png (+ .pdf) and fig3_effects_<a..g>.png,
         results/pancreas_corrected/fig3_values.csv, evidence/pancreas_condition_by_source.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.background(1)

import argparse                                            # noqa: E402
import math                                                # noqa: E402
import os                                                  # noqa: E402

import matplotlib                                          # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
CELLTYPES = ["Ductal cell", "Endothelial cell", "Stellate cell"]
REF, TEST = "Reference", "pancreas"
LN2 = math.log(2.0)
EPS = 1e-9
N_EACH = 20
METHODS = {  # label: (file, column transform, unit)
    "DUET": ("DUET", "log2 units: difference of log1p means"),
    "MAST": ("MAST", "log2 units: difference of log1p means"),
    "diffxpy": ("diffxpy", "log2 fold change (cell-level NB GLM)"),
    "DESeq2": ("DESeq2", "log2 fold change (pseudobulk NB GLM)"),
    "Wilcoxon": ("wilcoxon", "log2 ratio of back-transformed log means"),
}


def common_effects(h5ad: Path) -> pd.DataFrame:
    frames = []
    for ct in CELLTYPES:
        A = _runtime.load_rows(h5ad, lambda o: (o["celltype"].astype(str) == ct).to_numpy(), layers=())
        cond = A.obs["Sample"].astype(str).to_numpy()
        X = A.X.tocsr()
        row = {}
        for lab in (REF, TEST):
            m = cond == lab
            Xm = X[m]
            row[f"mean_{lab}"] = np.asarray(Xm.mean(axis=0)).ravel()
            row[f"det_{lab}"] = np.asarray((Xm > 0).mean(axis=0)).ravel()
        d = pd.DataFrame({"gene": A.var_names.astype(str), "celltype": ct,
                          "mean_ref": row[f"mean_{REF}"], "mean_test": row[f"mean_{TEST}"],
                          "det_ref": row[f"det_{REF}"], "det_test": row[f"det_{TEST}"]})
        d["common_log2fc"] = np.log2((np.expm1(d.mean_test) + EPS) / (np.expm1(d.mean_ref) + EPS))
        d["delta_det_pp"] = 100 * (d.det_test - d.det_ref)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def add_methods(E: pd.DataFrame, resdir: Path) -> pd.DataFrame:
    for label, (m, _) in METHODS.items():
        r = pd.read_csv(resdir / f"de_{m}_all_celltypes.csv", low_memory=False)
        r = r[r.tested == True].drop_duplicates(["gene", "celltype"])                   # noqa: E712
        E = E.merge(r[["gene", "celltype", "log2fc"]].rename(columns={"log2fc": f"lfc_{label}"}),
                    on=["gene", "celltype"], how="left")
    return E


def select_genes(E: pd.DataFrame) -> list[str]:
    ok = (E[["det_ref", "det_test"]].max(axis=1) >= 0.10)
    for label in METHODS:
        ok &= E[f"lfc_{label}"].notna()
    good = E[ok].groupby("gene")["celltype"].nunique()
    genes = good.index[good == len(CELLTYPES)]
    eff = E[E.gene.isin(genes)].groupby("gene")["common_log2fc"].mean().sort_values()
    return list(eff.tail(N_EACH).index[::-1]) + list(eff.head(N_EACH).index[::-1])   # up first


def condition_by_source(h5ad: Path) -> pd.DataFrame:
    obs = _runtime.read_obs(h5ad)
    acc = pd.read_csv(HERE / "evidence" / "additional_file_1_sample_accessions.csv")
    gsm2gse = dict(zip(acc.sample_accession.astype(str), acc.series.astype(str)))
    src = obs["donor"].astype(str)
    series = src.map(lambda s: "HPAP" if s.startswith("HPAP") else gsm2gse.get(s.split("_")[0], "?"))
    t = pd.DataFrame({"series": series, "condition": obs["Sample"].astype(str), "donor": src})
    out = t.groupby(["series", "condition"]).agg(donors=("donor", "nunique"), cells=("donor", "size")).reset_index()
    return out


def draw(ax, M: np.ndarray, title: str, unit: str, ylabels=None, fig=None):
    v = float(np.nanpercentile(np.abs(M), 99)) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xticks(range(len(CELLTYPES)))
    ax.set_xticklabels([c.replace(" cell", "") for c in CELLTYPES], rotation=45, ha="right", fontsize=10)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=10)
    else:
        ax.set_yticks([])
    ax.axhline(N_EACH - 0.5, color="k", lw=0.6)
    cb = (fig or ax.figure).colorbar(im, ax=ax, fraction=0.08, pad=0.03)
    cb.ax.tick_params(labelsize=9)
    cb.set_label(unit, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=str(INPUTS / "PancreasCorrected.h5ad"))
    ap.add_argument("--results-dir", default=str(HERE / "figure_data"))
    ap.add_argument("--figdir", default=str(HERE / "figures"))
    a = ap.parse_args()
    resdir, figdir = Path(a.results_dir), Path(a.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    S = pd.read_csv(resdir / "fig3_values.csv")
    genes = select_genes(S)

    def mat(col):
        return S.pivot(index="gene", columns="celltype", values=col).reindex(index=genes, columns=CELLTYPES).to_numpy()

    panels = [("(a) common log2FC", "common_log2fc", "log2 ratio of back-transformed log means"),
              ("(b) Δ detection", "delta_det_pp", "percentage points")]
    panels += [(f"({chr(ord('c') + i)}) {lab}", f"lfc_{lab}", unit) for i, (lab, (_, unit)) in enumerate(METHODS.items())]

    fig, axes = plt.subplots(1, len(panels), figsize=(12, 11), constrained_layout=True)
    for i, (ax, (title, col, unit)) in enumerate(zip(axes, panels)):
        draw(ax, mat(col), title, unit, ylabels=genes if i == 0 else None, fig=fig)
    fig.suptitle("Pancreas: common effect sizes (a, b) and each method's own output (c-g); "
                 "tumour vs non-tumour is confounded with data source", fontsize=9)
    fig.savefig(figdir / "fig3_effects.png", dpi=200)
    fig.savefig(figdir / "fig3_effects.pdf")
    plt.close(fig)
    for title, col, unit in panels:                           # one plot per image, each with its own scale
        f, ax = plt.subplots(figsize=(3.0, 9), constrained_layout=True)
        draw(ax, mat(col), title, unit, ylabels=genes)
        f.savefig(figdir / f"fig3_effects_{title[1]}.png", dpi=200)
        plt.close(f)
    print(f"[fig3] {len(genes)} genes; wrote {figdir / 'fig3_effects.png'} and {len(panels)} single panels")


if __name__ == "__main__":
    main()
