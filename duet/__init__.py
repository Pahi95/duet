"""DUET — Detection–Expression Unified Test.

A pure-Python two-part *hurdle* engine for single-cell differential expression:
a logistic **detection** model and a Gaussian **positive-expression** model,
combined by summing their likelihood-ratio statistics into a single chi-squared
test. Same high-level model as R MAST, implemented in NumPy/SciPy directly on
sparse AnnData -- no R toolchain, no dense materialization, no disk round-trip.

    from duet import run_duet
    out = run_duet(adata, output_dir="results", celltype_col="celltype",
                   sample_col="Sample", ref_label="Reference",
                   test_label="pancreas", mast_compat=True)

Ranking note: rank genes on ``neglog10p``, not ``pvalue``. At tens of thousands
of cells the hurdle statistic routinely pushes the true p-value below the
float64 floor, where ``pvalue`` stores exactly 0.0 and ties every such gene.

``run_scpyde``, ``run_scpyde_cnv`` and ``run_python_hurdle_de`` are legacy
aliases and continue to work.
"""

from .pseudobulk import (
    run_duet_pseudobulk,
    run_pseudobulk,
    aggregate_pseudobulk,
    combine_calls,
)
from .core import (
    run_duet,
    run_duet_cnv,
    run_scpyde,                # deprecated alias -> run_duet
    run_scpyde_cnv,            # deprecated alias -> run_duet_cnv
    run_python_hurdle_de,      # backward-compatible aliases
    run_python_hurdle_cnv,
    HURDLE_OUTPUT_COLUMNS,
    METHOD_NAME,
    __version__,
)

__all__ = [
    "run_duet_pseudobulk",
    "run_pseudobulk",
    "aggregate_pseudobulk",
    "combine_calls",
    "run_duet",
    "run_duet_cnv",
    "run_scpyde",
    "run_scpyde_cnv",
    "run_python_hurdle_de",
    "run_python_hurdle_cnv",
    "HURDLE_OUTPUT_COLUMNS",
    "METHOD_NAME",
    "__version__",
]
