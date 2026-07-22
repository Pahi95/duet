"""DEPRECATED compatibility shim — ``CustomDE`` is now :mod:`duet`.

The algorithm formerly branded **scPyDE** was renamed to **DUET**
(Detection–Expression Unified Test) in July 2026, because "scPyDE" read as
"Python SCDE" and SCDE (Kharchenko et al., Nat Methods 2014) is an established
but statistically unrelated method — a Bayesian dropout error model, not a
two-part hurdle.

This module re-exports the new package so existing callers keep working:

    from CustomDE.scpyde import run_scpyde     # old, still works, warns
    from duet import run_duet                  # new

Note that the ``method`` column in DUET's output is now ``"DUET"``, so result
filenames follow ``de_DUET_all_celltypes.csv``. Anything that greps for
``scPyDE`` in results needs updating; the import path alone does not cover it.
"""

import sys as _sys
import warnings as _warnings

from duet.core import (  # noqa: F401
    run_duet,
    run_duet_cnv,
    run_scpyde,
    run_scpyde_cnv,
    run_python_hurdle_de,
    run_python_hurdle_cnv,
    HURDLE_OUTPUT_COLUMNS,
    METHOD_NAME,
    __version__,
)
from duet import core as _core

_warnings.warn(
    "CustomDE is deprecated and will be removed: the package is now 'duet' and "
    "the method is named DUET (was scPyDE). Use `from duet import run_duet`.",
    DeprecationWarning,
    stacklevel=2,
)

# Keep `from CustomDE.scpyde import ...` resolvable, since scpyde.py is gone.
scpyde = _core
_sys.modules.setdefault(__name__ + ".scpyde", _core)

__all__ = [
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
