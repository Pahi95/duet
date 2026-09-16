#!/usr/bin/env python
"""
collect_evidence.py
===================
Phase 2.5 — consolidate every result into one citable evidence pack, and VERIFY
the headline numbers by re-deriving them from the per-method files rather than
copying them out of summaries.

The point is not to re-run anything. It is to make sure that what gets written in
the paper matches what the pipeline actually produced. Every claim carried
forward from the analysis is listed in CLAIMS below with the file it should come
from; each is recomputed and marked OK or MISMATCH. A number nobody can
re-derive is a number that should not be published.

Outputs (evidence/):
  environment.json        interpreter, package and R/MAST versions, machine
  datasets.csv            cells, genes, samples, design per dataset
  method_summary.csv      significant genes per method per dataset x cell type
  mast_agreement.csv      DUET vs R MAST on identical cells, all datasets
  benchmark.csv           runtime / memory scaling vs R MAST
  null_calibration.csv    both nulls x 3 methods x 3 datasets
  simulation.csv          AUC / power / observed FDR, and per-category detection
  claims_check.csv        every headline number, recomputed, OK or MISMATCH
  provenance.csv          every source file with size, mtime and row count
  EVIDENCE.md             human-readable digest of all of the above
"""
from __future__ import annotations
import json, os, platform, subprocess, sys, hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
# Primary data lives outside the checkout, in DUET/inputs.
# Override with the DUET_INPUTS environment variable.
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
OUT = HERE / "evidence"
METHODS = ["DUET", "MAST", "diffxpy", "DESeq2", "wilcoxon"]

DATASETS = {
    "pancreas": dict(dir="results/pancreas", h5ad="PancreasIntegratedAnnotated.h5ad",
                     ref="Reference", test="pancreas",
                     design="across-donor, condition confounded with donor"),
    "kang":     dict(dir="results/kang", h5ad="data/kang_8donors.h5ad",
                     ref="ctrl", test="stim",
                     design="paired within-donor, multiplexed pool"),
    "crowell":  dict(dir="results/crowell", h5ad="data/crowell_4vs4.h5ad",
                     ref="Vehicle", test="LPS",
                     design="across-sample, balanced 4v4"),
    "sim":      dict(dir="results/sim", h5ad="data/sim_muscat.h5ad",
                     ref="A", test="B",
                     design="muscat simulation from the Crowell reference, 4v4"),
}

# (label, expected, tolerance, how to recompute) -- see verify_claims()
CLAIMS = [
    ("DUET vs MAST log2FC Pearson, Kang CD4 T cells",        1.000,  0.002),
    ("DUET vs MAST top-100 overlap, Kang CD4 T cells",       100,    0),
    ("DUET vs MAST log2FC Pearson, Crowell Excit. Neuron",   1.000,  0.002),
    ("DUET vs MAST sig-gene Jaccard, Crowell Excit. Neuron", 1.000,  0.002),
    ("Simulation: DUET AUC",                                 0.9778, 0.002),
    ("Simulation: MAST AUC",                                 0.9776, 0.002),
    ("Simulation: DUET observed FDR",                        0.806,  0.005),
    ("Simulation: MAST observed FDR",                        0.805,  0.005),
    ("Simulation: DESeq2 observed FDR",                      0.000,  0.005),
    ("Simulation: DUET detection of bimodal (db) genes, %",  98.8,   0.2),
    ("Simulation: DESeq2 detection of bimodal (db) genes, %", 28.9,  0.2),
    ("Simulation: DUET false calls on true nulls (ee), %",   52.3,   0.2),
    ("Benchmark: MAST/DUET fit speed-up at 11106 cells",     117,    2),
    ("Benchmark: MAST peak RAM at 11106 cells, MB",          5175,   50),
    ("Null: DUET donor-swap, pancreas Ductal, %",            75.49,  0.2),
    ("Null: DUET donor-swap, Kang B cells, %",               1.00,   0.2),
    ("Null: DUET donor-swap, Crowell Excit. Neuron, %",      100.0,  0.2),
    ("Null: pseudobulk worst donor-swap across all, %",      0.85,   0.2),
    ("Null: DUET permutation type-I, max excl. pancreas Ductal", 0.0587, 0.002),
    ("MATCHED pancreas Ductal (11106 cells): DUET vs MAST log2FC Pearson", 1.000, 0.002),
    ("MATCHED pancreas Ductal (11106 cells): sign agreement, %",           100.0, 0.05),
    ("UNMATCHED pancreas Ductal (old pipeline CSV): log2FC Pearson",       0.9635, 0.002),
    ("Ductal permutation type-I: mean over 10 seeds",                      0.0627, 0.002),
    ("Ductal permutation type-I: mean EXCLUDING the seed-0 outlier",       0.0472, 0.002),
    ("Ductal permutation: seeds with 0 genes at FDR<0.05, of 10",          9,      0),
    ("Covariate path: DUET vs MAST log2FC Pearson",                       0.9994, 0.002),
    ("Covariate path: -log10p Spearman",                                  0.9998, 0.002),
    ("Scaling: speed-up at 23,895 cells (fit)",                           148.8,   6.0),
    ("Scaling: speed-up at 23,895 cells (end-to-end)",                    282.1,  10.0),
    ("Scaling: MAST peak GB at 23,895 cells",                               9.41,  0.15),
    ("Scaling: DUET seconds at 23,895 cells",                               3.36,  0.40),
    ("Scaling: MAST memory MB per cell",                                    0.368, 0.010),
    ("Scaling: MAST memory linearity r2",                                   1.000, 0.002),
    ("Built-in pseudobulk: donor-swap splits",                             20,     0),
    ("Built-in pseudobulk: cell-level mean, %",                            34.9,   0.5),
    ("Built-in pseudobulk: cell-level max, %",                             99.0,   0.5),
    ("Built-in pseudobulk: sample-level mean, %",                           0.04,  0.01),
    ("Built-in pseudobulk: sample-level max, %",                            0.24,  0.01),
    ("Built-in pseudobulk: sample-level splits above 5%",                   0,     0),
    ("Covariate timing: DUET numpy_irls seconds",                          78.3,   6.0),
    ("Covariate timing: MAST fit+test seconds",                            57.1,   5.0),
    ("Covariate timing: MAST/DUET ratio (numpy_irls)",                      1.37,  0.15),
    ("Covariate timing: engine speed-up statsmodels->numpy_irls",           2.80,  0.30),
    ("Engine equivalence: min significant-call Jaccard",                    1.0000, 0.0005),
    ("Engine equivalence: total tests compared",                        80947.0,   1.0),
    ("Donor-swap over 10 splits: DUET mean, Kang",                        24.57,  0.5),
    ("Donor-swap over 10 splits: DUET mean, pancreas",                    74.56,  0.5),
    ("Donor-swap over 10 splits: DUET mean, Crowell",                     83.43,  0.5),
    ("Donor-swap over 10 splits: pseudobulk max of ANY split, %",         2.47,   0.1),
    ("Donor-swap over 10 splits: pseudobulk splits above 5%, of 90",      0,      0),
    ("Sim replicates: DUET observed FDR mean over 5",                     0.3910, 0.005),
    ("Sim replicates: DUET observed FDR min",                             0.0469, 0.002),
    ("Sim replicates: DUET observed FDR max",                             0.8731, 0.002),
    ("Sim replicates: DUET replicates with FDR<=0.10, of 5",              2,      0),
    ("Sim replicates: pseudobulk FDR max over 5",                         0.0,    0.001),
    ("Sim replicates: max |DUET-MAST| FDR difference",                    0.0122, 0.002),
]


def _sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception:
        return ""


def _cpu_name() -> str:
    """Marketing name of the CPU (platform.processor() only gives the family on Windows)."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        return platform.processor() or "?"


def _blas() -> list[dict]:
    """Which BLAS/OpenMP libraries are loaded, and with how many threads."""
    try:
        import numpy  # noqa: F401  (loads its BLAS so threadpoolctl can see it)
        import scipy.linalg  # noqa: F401
        from threadpoolctl import threadpool_info
        keep = ("user_api", "internal_api", "version", "num_threads", "threading_layer", "filepath")
        return [{k: str(v) for k, v in lib.items() if k in keep} for lib in threadpool_info()]
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]


def _git() -> dict:
    head = _sh(["git", "-C", str(HERE), "rev-parse", "HEAD"])
    dirty = _sh(["git", "-C", str(HERE), "status", "--porcelain", "--untracked-files=no"])
    return {"commit": head or "?", "uncommitted_changes": bool(dirty),
            "changed_files": dirty.splitlines() if dirty else []}


def environment() -> dict:
    """Everything needed to rerun the analyses on a comparable machine.

    Simulation seeds are not machine properties; they are written next to each
    simulation's outputs by the scripts that generate them.
    """
    from importlib import metadata
    dist = {"sklearn": "scikit-learn"}
    pkgs = {}
    for m in ("numpy", "scipy", "pandas", "anndata", "scanpy", "sklearn", "statsmodels",
              "pydeseq2", "diffxpy", "matplotlib", "h5py", "mpmath", "psutil", "threadpoolctl"):
        try:
            pkgs[m] = metadata.version(dist.get(m, m))
        except metadata.PackageNotFoundError:
            pkgs[m] = "not installed"
    try:
        import duet
        pkgs["duet"] = getattr(duet, "__version__", "?")
    except Exception:
        pkgs["duet"] = "not importable"
    rscript = os.environ.get("RSCRIPT", r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe")
    rver = _sh([rscript, "-e", 'cat(R.version.string)']).splitlines()
    rpkgs = {}
    for p in ("MAST", "DESeq2", "muscat", "muscData", "SingleCellExperiment", "Matrix",
              "lme4", "glmmTMB", "GLIMES"):
        # GLIMES lives in the project-local library (DUET/_Rlib), next to the user library
        rlib = (HERE.parent / "_Rlib").as_posix()
        o = _sh([rscript, "-e", f'.libPaths(c("{rlib}", .libPaths())); '
                                f'cat(tryCatch(as.character(packageVersion("{p}")), '
                                f'error=function(e) "not installed"))']).splitlines()
        rpkgs[p] = o[-1] if o else "?"
    # sessionInfo() reports no BLAS path on Windows; these work on every platform.
    # An empty BLAS string means R's bundled reference BLAS (Rblas.dll) is in use.
    rblas = _sh([rscript, "-e", 'b <- extSoftVersion()[["BLAS"]]; '
                                'cat(paste("BLAS:", if (nzchar(b)) b else "R internal reference BLAS"), '
                                'paste("LAPACK:", La_version(), La_library()), sep="\\n")']).splitlines()
    mem_gb = None
    try:
        import psutil
        mem_gb = round(psutil.virtual_memory().total / 1e9, 1)
        cores = {"physical": psutil.cpu_count(logical=False), "logical": psutil.cpu_count(logical=True)}
    except Exception:
        cores = {"logical": os.cpu_count()}
    threads = {k: os.environ.get(k, "") for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                   "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    return dict(
        collected_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        python=sys.version.split()[0], platform=platform.platform(),
        processor=platform.processor() or "?", cpu=_cpu_name(), cpu_cores=cores, ram_gb=mem_gb,
        blas=_blas(), thread_env=threads, git=_git(), packages=pkgs,
        R=[l for l in rver if l.startswith("R version")][-1:] or ["?"],
        R_blas_lapack=[l for l in rblas if l.strip() and not l.startswith("[")],
        R_packages=rpkgs,
    )


def load(ds: str, method: str) -> pd.DataFrame | None:
    f = HERE / DATASETS[ds]["dir"] / f"de_{method}_all_celltypes.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f, low_memory=False)
    return d[d["tested"] == True]                                # noqa: E712


def method_summary() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for m in METHODS:
            d = load(ds, m)
            if d is None or not len(d):
                continue
            for ct, g in d.groupby("celltype"):
                rows.append(dict(dataset=ds, celltype=ct, method=m,
                                 n_tested=len(g),
                                 n_sig_fdr05=int((g["fdr"] < 0.05).sum()),
                                 pct_sig=round(100 * float((g["fdr"] < 0.05).mean()), 2)))
    return pd.DataFrame(rows)


def mast_agreement() -> pd.DataFrame:
    from scipy.stats import spearmanr
    rows = []
    for ds in DATASETS:
        a, b = load(ds, "DUET"), load(ds, "MAST")
        if a is None or b is None:
            continue
        for ct in sorted(set(a.celltype) & set(b.celltype)):
            x = a[a.celltype == ct].drop_duplicates("gene").set_index("gene")
            y = b[b.celltype == ct].drop_duplicates("gene").set_index("gene")
            g = x.index.intersection(y.index)
            if len(g) < 100:
                continue
            x, y = x.loc[g], y.loc[g]
            nx = x["neglog10p"] if "neglog10p" in x else -np.log10(x["pvalue"].clip(1e-300))
            ny = y["neglog10p"] if "neglog10p" in y else -np.log10(y["pvalue"].clip(1e-300))
            sx, sy = x["fdr"] < 0.05, y["fdr"] < 0.05
            tA = set(nx.nlargest(100).index); tB = set(ny.nlargest(100).index)
            # For pancreas, results/de_MAST_all_celltypes.csv is a MAST run from a
            # DIFFERENT pipeline pass on a different cell selection, so it measures
            # pipeline drift, not method agreement. The matched pancreas comparison
            # lives in bench/ and is appended separately below.
            rows.append(dict(
                dataset=ds, celltype=ct, n_genes=len(g),
                source=("pipeline CSV (UNMATCHED cells)" if ds == "pancreas"
                        else "run_mast_dataset (matched)"),
                matched_cells=(ds != "pancreas"),
                log2fc_pearson=round(float(x["log2fc"].corr(y["log2fc"])), 4),
                sign_agree_pct=round(100 * float((np.sign(x["log2fc"]) == np.sign(y["log2fc"])).mean()), 2),
                neglog10p_spearman=round(float(spearmanr(nx, ny).statistic), 4),
                top100_overlap=len(tA & tB),
                sig_jaccard=round(float((sx & sy).sum() / max((sx | sy).sum(), 1)), 4),
                sig_duet=int(sx.sum()), sig_mast=int(sy.sum())))

    # The MATCHED pancreas comparison: bench_mast_export.py exported DUET's exact
    # gene set for a given cell budget and bench_mast_run.R fitted MAST on it, so
    # these rows are directly comparable to the Kang/Crowell/sim rows above.
    bf = HERE / "bench" / "benchmark_summary.csv"
    if bf.is_file():
        b = pd.read_csv(bf).dropna(subset=["lfc_pearson"])
        for _, r in b.iterrows():
            rows.append(dict(
                dataset="pancreas", celltype=f"Ductal cell (n={int(r.n_cells)})",
                n_genes=int(r.n_compared), source="bench_mast_run (matched)",
                matched_cells=True,
                log2fc_pearson=round(float(r.lfc_pearson), 4),
                sign_agree_pct=round(float(r.sign_agree_pct), 2),
                neglog10p_spearman=round(float(r.neglog10p_spearman), 4),
                top100_overlap=np.nan,
                sig_jaccard=round(float(r.sig_jaccard), 4),
                sig_duet=int(r.sig_duet), sig_mast=int(r.sig_mast)))
    return pd.DataFrame(rows)


def simulation() -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = pd.read_csv(HERE / "results" / "sim" / "sim_evaluation.csv")
    cat = pd.read_csv(HERE / "results" / "sim" / "sim_by_category.csv")
    return ev, cat


def nulls() -> pd.DataFrame:
    fs = sorted((HERE / "results" / "null").glob("null_*_matched.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def benchmark() -> pd.DataFrame:
    f = HERE / "bench" / "benchmark_summary.csv"
    if not f.is_file():
        return pd.DataFrame()
    t = pd.read_csv(f)
    t = t.dropna(subset=["fit_s"]).copy()
    t["mast_end2end_s"] = t.read_s + t.fit_s + t.lrt_s
    t["speedup_fit"] = (t.fit_s / t.duet_fit_s).round(1)
    t["speedup_e2e"] = (t.mast_end2end_s / t.duet_fit_s).round(1)
    return t[["tag", "n_cells", "n_genes", "duet_fit_s", "fit_s", "lrt_s",
              "mast_end2end_s", "speedup_fit", "speedup_e2e", "peak_mb"]]


def verify_claims(agr, ev, cat, nul, ben) -> pd.DataFrame:
    """Recompute each headline number and compare with what has been claimed."""
    got = {}

    def g(df, q, col):
        s = df.query(q)
        return float(s[col].iloc[0]) if len(s) else np.nan

    if len(agr):
        got["DUET vs MAST log2FC Pearson, Kang CD4 T cells"] = g(
            agr, "dataset=='kang' and celltype=='CD4 T cells'", "log2fc_pearson")
        got["DUET vs MAST top-100 overlap, Kang CD4 T cells"] = g(
            agr, "dataset=='kang' and celltype=='CD4 T cells'", "top100_overlap")
        got["DUET vs MAST log2FC Pearson, Crowell Excit. Neuron"] = g(
            agr, "dataset=='crowell' and celltype=='Excit. Neuron'", "log2fc_pearson")
        got["DUET vs MAST sig-gene Jaccard, Crowell Excit. Neuron"] = g(
            agr, "dataset=='crowell' and celltype=='Excit. Neuron'", "sig_jaccard")
    if len(ev):
        for m in ("DUET", "MAST"):
            got[f"Simulation: {m} AUC"] = g(ev, f"method=='{m}'", "auc")
            got[f"Simulation: {m} observed FDR"] = g(ev, f"method=='{m}'", "observed_fdr")
        got["Simulation: DESeq2 observed FDR"] = g(ev, "method=='DESeq2'", "observed_fdr")
    if len(cat):
        got["Simulation: DUET detection of bimodal (db) genes, %"] = g(cat, "category=='db'", "DUET")
        got["Simulation: DESeq2 detection of bimodal (db) genes, %"] = g(cat, "category=='db'", "DESeq2")
        got["Simulation: DUET false calls on true nulls (ee), %"] = g(cat, "category=='ee'", "DUET")
    if len(ben):
        b = ben[ben.n_cells == 11106]
        if len(b):
            got["Benchmark: MAST/DUET fit speed-up at 11106 cells"] = float(b.speedup_fit.iloc[0])
            got["Benchmark: MAST peak RAM at 11106 cells, MB"] = float(b.peak_mb.iloc[0])
    if len(nul):
        d = nul[nul.design == "donor_swap"]
        got["Null: DUET donor-swap, pancreas Ductal, %"] = g(
            d, "dataset=='pancreas' and method=='DUET' and celltype.str.startswith('Ductal')",
            "pct_sig_fdr05")
        got["Null: DUET donor-swap, Kang B cells, %"] = g(
            d, "dataset=='kang' and method=='DUET' and celltype.str.startswith('B')",
            "pct_sig_fdr05")
        got["Null: DUET donor-swap, Crowell Excit. Neuron, %"] = g(
            d, "dataset=='crowell' and method=='DUET' and celltype.str.startswith('Excit')",
            "pct_sig_fdr05")
        pb = d[d.method.str.contains("pseudobulk", case=False, na=False)]
        got["Null: pseudobulk worst donor-swap across all, %"] = (
            float(pb.pct_sig_fdr05.max()) if len(pb) else np.nan)
        perm = nul[(nul.design == "permutation") & (nul.method == "DUET")].copy()
        perm = perm[~((perm.dataset == "pancreas") & perm.celltype.str.startswith("Ductal"))]
        got["Null: DUET permutation type-I, max excl. pancreas Ductal"] = (
            float(perm.type1_at_alpha05.max()) if len(perm) else np.nan)
    if len(agr):
        m = agr.query("dataset=='pancreas' and source=='bench_mast_run (matched)' "
                      "and celltype.str.contains('11106')")
        if len(m):
            got["MATCHED pancreas Ductal (11106 cells): DUET vs MAST log2FC Pearson"] = float(m.log2fc_pearson.iloc[0])
            got["MATCHED pancreas Ductal (11106 cells): sign agreement, %"] = float(m.sign_agree_pct.iloc[0])
        u = agr.query("dataset=='pancreas' and matched_cells==False and celltype=='Ductal cell'")
        if len(u):
            got["UNMATCHED pancreas Ductal (old pipeline CSV): log2FC Pearson"] = float(u.log2fc_pearson.iloc[0])
    sd = OUT / "ductal_permutation_seeds.csv"
    if sd.is_file():
        t = pd.read_csv(sd)
        got["Ductal permutation type-I: mean over 10 seeds"] = float(t.type1_hurdle.mean())
        got["Ductal permutation type-I: mean EXCLUDING the seed-0 outlier"] = float(
            t[t.seed != 0].type1_hurdle.mean())
        got["Ductal permutation: seeds with 0 genes at FDR<0.05, of 10"] = float(
            (t.n_sig_fdr05 == 0).sum())

    cf = HERE / "bench" / "covariate" / "covariate_agreement.csv"
    if cf.is_file():
        c = pd.read_csv(cf).iloc[0]
        got["Covariate path: DUET vs MAST log2FC Pearson"] = float(c.log2fc_pearson)
        got["Covariate path: -log10p Spearman"] = float(c.neglog10p_spearman)

    # Extended scaling ladder on a POOLED population (Ductal+Endothelial+Stellate),
    # 2,500-23,895 cells. Cost benchmark only: the arms are not compositionally
    # matched, so no agreement statistic is taken from it.
    sf = HERE / "bench" / "pooled_scaling.csv"
    if sf.is_file():
        sc = pd.read_csv(sf).sort_values("cells")
        top = sc.iloc[-1]
        got["Scaling: speed-up at 23,895 cells (fit)"] = float(top.x_fit)
        got["Scaling: speed-up at 23,895 cells (end-to-end)"] = float(top.x_e2e)
        got["Scaling: MAST peak GB at 23,895 cells"] = float(top.peak_gb)
        got["Scaling: DUET seconds at 23,895 cells"] = float(top.duet)
        got["Scaling: MAST memory MB per cell"] = float(
            np.polyfit(sc.cells, sc.peak_gb * 1024, 1)[0])
        got["Scaling: MAST memory linearity r2"] = float(
            np.corrcoef(sc.cells, sc.peak_gb)[0, 1] ** 2)

    # The built-in sample-level arm (run_duet_pseudobulk), measured through the
    # SHIPPED entry point rather than the benchmark harness. These are the numbers
    # the manuscript quotes for the combined caller.
    pbf = HERE / "evidence" / "pseudobulk_null.csv"
    if pbf.is_file():
        pn = pd.read_csv(pbf)
        got["Built-in pseudobulk: donor-swap splits"] = float(len(pn))
        got["Built-in pseudobulk: cell-level mean, %"] = float(pn.duet_pct.mean())
        got["Built-in pseudobulk: cell-level max, %"] = float(pn.duet_pct.max())
        got["Built-in pseudobulk: sample-level mean, %"] = float(pn.pseudobulk_pct.mean())
        got["Built-in pseudobulk: sample-level max, %"] = float(pn.pseudobulk_pct.max())
        got["Built-in pseudobulk: sample-level splits above 5%"] = float(
            (pn.pseudobulk_pct > 5).sum())

    # Covariate TIMING, measured isolated and best-of-N. The earlier figures
    # (DUET 210 s, MAST 118 s) were collected under unknown load; MAST in
    # particular re-measures at 57 s, so the ratio -- not just the absolutes --
    # was wrong.
    tf = HERE / "bench" / "covariate_timing" / "covariate_timing.csv"
    if tf.is_file():
        t = pd.read_csv(tf).set_index("method")
        sm = float(t.loc["DUET (statsmodels)", "best"])
        ni = float(t.loc["DUET (numpy_irls)", "best"])
        ma = float(t.loc["MAST (zlm+lrTest)", "best"])
        got["Covariate timing: DUET numpy_irls seconds"] = ni
        got["Covariate timing: MAST fit+test seconds"] = ma
        got["Covariate timing: MAST/DUET ratio (numpy_irls)"] = ni / ma
        got["Covariate timing: engine speed-up statsmodels->numpy_irls"] = sm / ni

    # Engine equivalence across datasets: how many gene-by-celltype tests differ,
    # and do the significant-call sets match?
    eng = sorted((HERE / "bench").glob("engine_*/engine_agreement.csv"))
    if eng:
        e = pd.concat([pd.read_csv(f) for f in eng], ignore_index=True)
        e = e[e.n_genes_tested > 0]
        if len(e):
            got["Engine equivalence: min significant-call Jaccard"] = float(e.sig_jaccard.min())
            got["Engine equivalence: total tests compared"] = float(e.n_genes_tested.sum())
    reps = sorted((HERE / "results" / "null").glob("null_reps_*.csv"))
    if reps:
        rr = pd.concat([pd.read_csv(f) for f in reps], ignore_index=True)
        du = rr[rr.method == "DUET"]
        for ds, key in (("kang", "Kang"), ("pancreas", "pancreas"), ("crowell", "Crowell")):
            v = du[du.dataset == ds]["pct_sig_fdr05"]
            if len(v):
                got[f"Donor-swap over 10 splits: DUET mean, {key}"] = float(v.mean())
        pb = rr[rr.method.str.contains("pseudobulk", case=False, na=False)]
        if len(pb):
            got["Donor-swap over 10 splits: pseudobulk max of ANY split, %"] = float(
                pb.pct_sig_fdr05.max())
            got["Donor-swap over 10 splits: pseudobulk splits above 5%, of 90"] = float(
                (pb.pct_sig_fdr05 > 5).sum())

    sr = HERE / "results" / "sim_reps" / "sim_replicates.csv"
    if sr.is_file():
        rp = pd.read_csv(sr)
        du = rp[rp.method == "DUET"]["observed_fdr"]
        got["Sim replicates: DUET observed FDR mean over 5"] = float(du.mean())
        got["Sim replicates: DUET observed FDR min"] = float(du.min())
        got["Sim replicates: DUET observed FDR max"] = float(du.max())
        got["Sim replicates: DUET replicates with FDR<=0.10, of 5"] = float((du <= 0.10).sum())
        pb = rp[rp.method == "DESeq2"]["observed_fdr"]
        got["Sim replicates: pseudobulk FDR max over 5"] = float(pb.max())
        piv = rp.pivot_table(index="replicate", columns="method", values="observed_fdr")
        if {"DUET", "MAST"} <= set(piv.columns):
            got["Sim replicates: max |DUET-MAST| FDR difference"] = float(
                (piv["DUET"] - piv["MAST"]).abs().max())

    rows = []
    for label, expected, tol in CLAIMS:
        actual = got.get(label, np.nan)
        if np.isnan(actual):
            status = "NOT RECOMPUTED"
        elif abs(actual - expected) <= tol:
            status = "OK"
        else:
            status = "*** MISMATCH ***"
        rows.append(dict(claim=label, claimed=expected, recomputed=actual,
                         tolerance=tol, status=status))
    return pd.DataFrame(rows)


def provenance() -> pd.DataFrame:
    rows = []
    for d in ("results", "results/kang", "results/crowell", "results/sim",
              "results/null", "bench"):
        p = HERE / d
        if not p.is_dir():
            continue
        for f in sorted(p.rglob("*")):
            if not f.is_file() or f.suffix not in (".csv", ".txt", ".json", ".png"):
                continue
            n = np.nan
            if f.suffix == ".csv":
                try:
                    n = sum(1 for _ in f.open(encoding="utf-8", errors="replace")) - 1
                except Exception:
                    pass
            rows.append(dict(path=str(f.relative_to(HERE)).replace("\\", "/"),
                             size_kb=round(f.stat().st_size / 1024, 1),
                             modified=datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
                             rows=n))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    print("[evidence] environment ...", flush=True)
    env = environment()
    (OUT / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    print("[evidence] method summary ...", flush=True)
    ms = method_summary(); ms.to_csv(OUT / "method_summary.csv", index=False)
    print("[evidence] MAST agreement ...", flush=True)
    agr = mast_agreement(); agr.to_csv(OUT / "mast_agreement.csv", index=False)
    print("[evidence] benchmark ...", flush=True)
    ben = benchmark(); ben.to_csv(OUT / "benchmark.csv", index=False)
    print("[evidence] nulls ...", flush=True)
    nul = nulls(); nul.to_csv(OUT / "null_calibration.csv", index=False)
    print("[evidence] simulation ...", flush=True)
    ev, cat = simulation()
    ev.to_csv(OUT / "simulation.csv", index=False)
    cat.to_csv(OUT / "simulation_by_category.csv", index=False)

    print("[evidence] verifying claims ...", flush=True)
    chk = verify_claims(agr, ev, cat, nul, ben)
    chk.to_csv(OUT / "claims_check.csv", index=False)
    prov = provenance(); prov.to_csv(OUT / "provenance.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 92)
    print("CLAIMS CHECK — every headline number, recomputed from the per-method files")
    print("=" * 92)
    print(chk.to_string(index=False))
    bad = chk[chk.status != "OK"]
    print(f"\n{len(chk) - len(bad)}/{len(chk)} verified OK")
    if len(bad):
        print("!! NEEDS ATTENTION:")
        print(bad.to_string(index=False))

    print("\n=== DUET vs R MAST, identical cells ===")
    print(agr.to_string(index=False))

    write_digest(env, ms, agr, ben, nul, ev, cat, chk, prov)
    print(f"\n[evidence] pack written to {OUT}")


def write_digest(env, ms, agr, ben, nul, ev, cat, chk, prov):
    L = []
    A = L.append
    A("# DUET — evidence pack\n")
    A(f"Collected {env['collected_utc']} · Python {env['python']} · {env['platform']}\n")
    A(f"R: {env['R'][0]} · MAST {env['R_packages'].get('MAST')} · "
      f"muscat {env['R_packages'].get('muscat')}\n")
    A("\nEvery number below is recomputed by `collect_evidence.py` from the "
      "per-method result files, not copied from a summary.\n")

    A("\n## Claim verification\n")
    ok = int((chk.status == "OK").sum())
    A(f"**{ok}/{len(chk)} headline numbers verified.**\n\n")
    A(chk.to_markdown(index=False))

    A("\n\n## Datasets\n")
    A(pd.DataFrame([dict(dataset=k, design=v["design"], ref=v["ref"], test=v["test"],
                         h5ad=v["h5ad"]) for k, v in DATASETS.items()]).to_markdown(index=False))

    A("\n\n## DUET vs R MAST\n")
    A("`matched_cells=True` means MAST was run by `run_mast_dataset.py` on exactly "
      "the same cells, genes and design (`~1+condition`). **The pancreas rows are "
      "`False`**: that MAST output came from a different pipeline run on a different "
      "cell selection, so those rows measure pipeline drift as much as method "
      "agreement and must not be quoted as the agreement figure. Cite the matched "
      "rows — Kang, Crowell and the simulation.\n\n")
    A(agr.to_markdown(index=False))

    A("\n\n## Runtime and memory vs R MAST\n")
    A(ben.to_markdown(index=False) if len(ben) else "_not collected_")

    A("\n\n## Ground truth (muscat simulation)\n")
    A(ev.to_markdown(index=False))
    A("\n\n### Detection by true category (% called at FDR<0.05)\n")
    A(cat.to_markdown(index=False))

    A("\n\n## Null calibration\n")
    if len(nul):
        piv = nul.pivot_table(index=["dataset", "celltype"], columns=["design", "method"],
                              values="pct_sig_fdr05")
        A(piv.to_markdown())

    A("\n\n## Significant genes per method\n")
    A(ms.pivot_table(index=["dataset", "celltype"], columns="method",
                     values="n_sig_fdr05").to_markdown())

    A(f"\n\n## Provenance\n\n{len(prov)} result files tracked; see provenance.csv "
      "for sizes, row counts and modification times.\n")
    (OUT / "EVIDENCE.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
