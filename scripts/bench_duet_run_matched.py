"""Future covariate timing driver with explicit EB and gating settings.

Same CLI as bench_duet_run.py. Use a NEW work directory. This driver was not
used to produce the historical CDR timings in V4 Table S5; no corrected timing
measurements are claimed in V4. Pair with MAST bayesglm, ebayes=TRUE, the same
CDR and condition design and identical input, CPU/thread and process settings.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from duet import core
import bench_duet_run

original=core.run_duet


def matched_run(*args,**kwargs):
    kwargs.update(min_cells_per_group=1,min_total_cells=20,min_positive=2,
                  partial_hurdle=True,eb_shrinkage=True)
    return original(*args,**kwargs)


if __name__=='__main__':
    core.run_duet=matched_run
    print('[configuration] matched-v4: min_positive=2 partial_hurdle=True eb_shrinkage=True',flush=True)
    bench_duet_run.main()
