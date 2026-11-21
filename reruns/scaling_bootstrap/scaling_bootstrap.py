"""Cell-level paired bootstrap of MAE_Q model differences under different SOH->Ah conversions.

Inputs (read-only): reruns/primary_benchmark/soh_primary_predictions.csv and
soh_primary_cell_metrics.csv from the 2026-11-06 rerun.

Conversions of the same predicted SOH values:
  OWN    predicted SOH x held-out cell's own Qref (pipeline method)
  TRAIN  predicted SOH x median Qref of the training-fold cells (fixed per fold;
         the bootstrap is conditional on the fitted folds and is not re-estimated)
  NOM    predicted SOH x nominal capacity (only needed for the XJTU first_comparable check)

Statistic (as in tools/reproduce_s6_s7.py): median(model A cell MAE_Q) - median(model B cell
MAE_Q); 5,000 Cell_ID resamples, seed 20260928, cells in lexicographic Cell_ID order, linear
percentile 95% intervals. Within a dataset every setting (Qref x conversion x model) uses the
same resampled cells. Each dataset draws from default_rng(20260928) with its own cell count, so
XJTU (55 cells) uses exactly the draws of S6/S7.
All outputs go to this script's directory.
"""
from pathlib import Path
from itertools import permutations
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRED = ROOT / 'reruns/primary_benchmark/soh_primary_predictions.csv'
CELLM = ROOT / 'reruns/primary_benchmark/soh_primary_cell_metrics.csv'
OUTPUTS = ['cell_mae_q.csv', 'model_pair_ci.csv', 'rank_frequency.csv', 'measured_minus_nominal_ci.csv',
           'reversal_assessment.csv', 'run_info.json']
if any((HERE / o).exists() for o in OUTPUTS):
    sys.exit('outputs exist; refusing to overwrite')

SEED, B = 20260928, 5000
NOMINAL = {'XJTU': 2.0, 'HUST': 1.1}
QREFS = ['nominal', 'first_comparable', 'early_median5_comparable']
QLABEL = {'nominal': 'nominal', 'first_comparable': 'first', 'early_median5_comparable': 'first5'}
MODELS = ['Random Forest Regressor', 'Gradient Boosting Regressor', 'Ridge Regression']
SHORT = dict(zip(MODELS, ['RF', 'GB', 'Ridge']))
METHODS = ['OWN', 'TRAIN', 'NOM']
PAIRS = [('RF', 'GB'), ('RF', 'Ridge'), ('GB', 'Ridge')]

p = pd.read_csv(PRED, encoding='utf-8-sig', low_memory=False,
                usecols=['Analysis_ID', 'Dataset', 'Cell_ID', 'Fold_ID', 'Qref_Definition', 'Qref_Ah', 'Model',
                         'Predicted_SOH_Fraction', 'Observed_Capacity_Ah'])
p = p[p.Analysis_ID.str.contains('PRIMARY_COMMON')].copy()
cellq = p.drop_duplicates(['Dataset', 'Qref_Definition', 'Cell_ID'])
train_med = {(d, q, k): float(g.loc[g.Fold_ID != k, 'Qref_Ah'].median())
             for (d, q), g in cellq.groupby(['Dataset', 'Qref_Definition']) for k in range(1, 6)}
conv = {'OWN': p.Qref_Ah,
        'TRAIN': pd.Series([train_med[t] for t in zip(p.Dataset, p.Qref_Definition, p.Fold_ID)], index=p.index),
        'NOM': p.Dataset.map(NOMINAL)}
cell_rows = []
for m, scale in conv.items():
    err = (p.Predicted_SOH_Fraction * scale - p.Observed_Capacity_Ah).abs()
    c = err.groupby([p.Dataset, p.Qref_Definition, p.Model, p.Cell_ID]).mean().rename('MAE_Q_Ah').reset_index()
    c['Method'] = m
    cell_rows.append(c)
cell = pd.concat(cell_rows, ignore_index=True)
cell.to_csv(HERE / 'cell_mae_q.csv', index=False, encoding='utf-8-sig')

# OWN must reproduce the pipeline's own per-cell MAE_Q.
cm = pd.read_csv(CELLM, encoding='utf-8-sig')
cm = cm[cm.Analysis_ID.str.contains('PRIMARY_COMMON')]
chk = cell[cell.Method == 'OWN'].merge(cm[['Dataset', 'Qref_Definition', 'Model', 'Cell_ID', 'MAE_Q_Ah']],
                                       on=['Dataset', 'Qref_Definition', 'Model', 'Cell_ID'], suffixes=('', '_pipeline'))
own_vs_pipeline = float((chk.MAE_Q_Ah - chk.MAE_Q_Ah_pipeline).abs().max())
assert len(chk) == len(cm), 'cell metric rows did not align'


def ci(x):
    lo, hi = np.quantile(x, [0.025, 0.975], method='linear')
    return float(lo), float(hi)


pair_rows, rank_rows, mm_rows, info = [], [], [], {'own_vs_pipeline_cell_mae_q_max_abs_diff': own_vs_pipeline}
for ds in ['XJTU', 'HUST']:
    sub = cell[cell.Dataset == ds]
    cells = sorted(sub.Cell_ID.unique())
    n = len(cells)
    # values[cell, qref, method, model]
    pv = sub.pivot_table(index='Cell_ID', columns=['Qref_Definition', 'Method', 'Model'], values='MAE_Q_Ah')
    values = np.stack([np.stack([np.stack([pv[(q, m, mod)].reindex(cells).to_numpy() for mod in MODELS], axis=-1)
                                 for m in METHODS], axis=1) for q in QREFS], axis=1)
    assert values.shape == (n, 3, 3, 3) and np.isfinite(values).all()
    draws = np.random.default_rng(SEED).integers(0, n, size=(B, n), dtype=np.int64)
    info[f'{ds}_cells'] = n
    info[f'{ds}_draw_sha256'] = hashlib.sha256(draws.astype('<i8', copy=False).tobytes()).hexdigest()
    obs = np.median(values, axis=0)              # (qref, method, model)
    boot = np.median(values[draws], axis=1)      # (B, qref, method, model)
    idx = {s: i for i, s in enumerate(['RF', 'GB', 'Ridge'])}
    for qi, q in enumerate(QREFS):
        for mi, meth in enumerate(METHODS):
            for a, b in PAIRS:
                d = boot[:, qi, mi, idx[a]] - boot[:, qi, mi, idx[b]]
                lo, hi = ci(d)
                pair_rows.append({'Dataset': ds, 'Qref': QLABEL[q], 'Method': meth, 'Pair': f'{a}-{b}',
                                  'Observed_Difference_Of_Medians_Ah': float(obs[qi, mi, idx[a]] - obs[qi, mi, idx[b]]),
                                  'Bootstrap_Median_Difference_Ah': float(np.median(d)),
                                  'CI95_Lower_Ah': lo, 'CI95_Upper_Ah': hi, 'CI_Includes_Zero': 'YES' if lo <= 0 <= hi else 'NO',
                                  f'{a}_Lower_Frequency': float(np.mean(d < 0)), f'{b}_Lower_Frequency': float(np.mean(d > 0)),
                                  'Tie_Frequency': float(np.mean(d == 0))})
            order = np.argsort(boot[:, qi, mi, :], axis=1, kind='stable')
            labels = np.array([' < '.join(['RF', 'GB', 'Ridge'][j] for j in row) for row in order])
            obs_order = ' < '.join(['RF', 'GB', 'Ridge'][j] for j in np.argsort(obs[qi, mi], kind='stable'))
            for perm in permutations(['RF', 'GB', 'Ridge']):
                lab = ' < '.join(perm)
                rank_rows.append({'Dataset': ds, 'Qref': QLABEL[q], 'Method': meth, 'Ranking': lab,
                                  'Frequency': float(np.mean(labels == lab)), 'Observed_Ranking': lab == obs_order})
    # Measured reference minus nominal, per model and conversion (same cells across Qrefs).
    for mi, meth in enumerate(METHODS):
        for mod in MODELS:
            k = MODELS.index(mod)
            for qi, q in enumerate(QREFS[1:], 1):
                d = boot[:, qi, mi, k] - boot[:, 0, mi, k]
                lo, hi = ci(d)
                mm_rows.append({'Dataset': ds, 'Method': meth, 'Model': SHORT[mod], 'Measured_Qref': QLABEL[q],
                                'Observed_Measured_Minus_Nominal_Ah': float(obs[qi, mi, k] - obs[0, mi, k]),
                                'CI95_Lower_Ah': lo, 'CI95_Upper_Ah': hi, 'CI_Includes_Zero': 'YES' if lo <= 0 <= hi else 'NO',
                                'Measured_Higher_Frequency': float(np.mean(d > 0))})

pairs = pd.DataFrame(pair_rows)
pairs.to_csv(HERE / 'model_pair_ci.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(rank_rows).to_csv(HERE / 'rank_frequency.csv', index=False, encoding='utf-8-sig')
mm = pd.DataFrame(mm_rows)
mm.to_csv(HERE / 'measured_minus_nominal_ci.csv', index=False, encoding='utf-8-sig')

# Reversal assessment: supported only if the reference conversion (OWN) and the alternative
# conversion both have RF-GB intervals excluding zero with opposite signs.
rev = []
for ds, q, alt in [('XJTU', 'first5', 'TRAIN'), ('HUST', 'first', 'TRAIN'), ('XJTU', 'first', 'NOM')]:
    r = {m: pairs[(pairs.Dataset == ds) & (pairs.Qref == q) & (pairs.Method == m) & (pairs.Pair == 'RF-GB')].iloc[0]
         for m in ['OWN', alt]}
    sign = {m: (np.sign(r[m].CI95_Lower_Ah) if r[m].CI_Includes_Zero == 'NO' else 0) for m in r}
    rev.append({'Dataset': ds, 'Qref': q, 'Alternative_Method': alt,
                'OWN_RF_minus_GB': r['OWN'].Observed_Difference_Of_Medians_Ah,
                'OWN_CI': f"[{r['OWN'].CI95_Lower_Ah:.5f}, {r['OWN'].CI95_Upper_Ah:.5f}]",
                'ALT_RF_minus_GB': r[alt].Observed_Difference_Of_Medians_Ah,
                'ALT_CI': f"[{r[alt].CI95_Lower_Ah:.5f}, {r[alt].CI95_Upper_Ah:.5f}]",
                'Supported_Reversal': 'YES' if sign['OWN'] != 0 and sign[alt] != 0 and sign['OWN'] == -sign[alt] else 'NO'})
rev = pd.DataFrame(rev)
rev.to_csv(HERE / 'reversal_assessment.csv', index=False, encoding='utf-8-sig')
info.update({'seed': SEED, 'replicates': B, 'statistic': 'median(model A cell MAE_Q) - median(model B cell MAE_Q)',
             'interval': 'linear percentile 2.5/97.5', 'cell_order': 'lexicographic Cell_ID',
             'numpy': np.__version__, 'pandas': pd.__version__})
(HERE / 'run_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20); pd.set_option('display.max_rows', 200)
print(json.dumps(info, indent=1))
cols = ['Dataset', 'Qref', 'Method', 'Pair', 'Observed_Difference_Of_Medians_Ah', 'CI95_Lower_Ah', 'CI95_Upper_Ah', 'CI_Includes_Zero']
print(pairs[cols].round(5).to_string(index=False))
rf = pd.DataFrame(rank_rows)
print(rf[rf.Frequency > 0].sort_values(['Dataset', 'Qref', 'Method', 'Frequency'], ascending=[True, True, True, False]).round(4).to_string(index=False))
print(mm.round(5).to_string(index=False))
print(rev.to_string(index=False))
