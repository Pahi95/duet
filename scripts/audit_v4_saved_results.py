"""Reanalyse saved simulation scores; never fit models or regenerate simulations.

Run: python scripts/audit_v4_saved_results.py --inputs ../inputs --output results/audit
Requires the archived results/sim_reps100/calls and original muscat gene_info.tsv files.
Calibration thresholds are read unchanged from fdr_matched.csv (replicates 1--50).
Category sensitivities are evaluated only on replicates 51--100 and the original
five-method common tested-gene universe. Bootstrap CIs resample whole replicates.
"""
from pathlib import Path
import argparse
import ast
import numpy as np
import pandas as pd

METHODS = {'DUET':'duet', 'MAST':'mast', 'wilcoxon':'wilcoxon',
           'diffxpy':'diffxpy', 'DESeq2':'deseq2'}


def run(root, inputs, output):
    output.mkdir(parents=True, exist_ok=True)
    saved = root / 'results' / 'sim_reps100'
    thresholds = pd.read_csv(saved / 'fdr_matched.csv').set_index('method')
    rows = []
    for rep in range(51, 101):
        name = f'sim_rep{rep:02d}'
        truth = pd.read_csv(inputs / 'data' / name / 'gene_info.tsv', sep='\t').set_index('gene')
        frames = {m: pd.read_csv(saved / 'calls' / name / f'{f}.csv').set_index('gene')
                  for m, f in METHODS.items()}
        universe = set(truth.index)
        for d in frames.values():
            universe &= set(d.index[d.tested.astype(str).str.lower().isin(['true','1']) & d.pvalue.notna()])
        genes = sorted(universe)
        categories = truth.loc[genes, 'category']
        for method, d in frames.items():
            called = d.loc[genes, 'score'] >= thresholds.loc[method, 'score_threshold']
            for category in ['de','dp','dm','db','ee','ep']:
                mask = categories == category
                n = int(mask.sum())
                k = int(called[mask].sum())
                rows.append(dict(replicate=rep, method=method, category=category,
                                 common_genes=len(genes), category_genes=n, called=k,
                                 call_fraction=k/n if n else np.nan))
    per = pd.DataFrame(rows)
    per.to_csv(output / 'category_calibrated_per_replicate.csv', index=False)
    summaries = []
    for (method, category), d in per.groupby(['method','category'], sort=False):
        x = d.call_fraction.dropna().to_numpy()
        rng = np.random.default_rng(0)
        ci = np.quantile(rng.choice(x, (2000,len(x)), replace=True).mean(axis=1), [.025,.975])
        summaries.append(dict(method=method, category=category, n=len(x), mean=x.mean(),
                              ci_low=ci[0], ci_high=ci[1],
                              interpretation='sensitivity' if category in ['de','dp','dm','db'] else 'false-positive proportion'))
    pd.DataFrame(summaries).to_csv(output / 'category_calibrated_summary.csv', index=False)
    cirows=[]
    for m, r in thresholds.iterrows():
        for metric in ['fdr','sensitivity']:
            lo,hi=ast.literal_eval(r[f'eval_{metric}_ci'])
            cirows.append(dict(method=m,metric=metric,mean=r[f'eval_{metric}_mean'],ci_low=lo,ci_high=hi,n=50))
    pd.DataFrame(cirows).to_csv(output / 'calibrated_confidence_intervals.csv',index=False)
    nullrows=[]
    for p in sorted((root/'results/null/donor_swap').glob('*.csv')):
        d=pd.read_csv(p)
        if 'n_discoveries' not in d: continue
        for (dataset,celltype,method),g in d.groupby(['dataset','celltype','method']):
            nullrows.append(dict(dataset=dataset,celltype=celltype,method=method,
                partitions=len(g),partitions_with_discovery=int((g.n_discoveries>0).sum()),
                share_with_discovery=float((g.n_discoveries>0).mean()),
                mean_discoveries=float(g.n_discoveries.mean()),
                mean_rejection_pct=float(g.rejection_pct.mean())))
    pd.DataFrame(nullrows).to_csv(output/'donor_partition_summary.csv',index=False)
    print(f'Wrote {len(per)} category rows and {len(nullrows)} donor-null summaries; no model fitting.')


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--results-root',type=Path,default=Path(__file__).resolve().parents[1],
                    help='Directory containing results/; useful for an extracted evidence archive.')
    args=ap.parse_args()
    run(args.results_root, args.inputs, args.output)
