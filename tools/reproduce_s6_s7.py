"""Reproduce FINAL S6/S7 from frozen A8 per-cell metrics; never modifies source results.

S6 statistic: median(GB cell MAE_Q) - median(RF cell MAE_Q).
Uses the same Cell_ID bootstrap draws across Qref definitions, seed 20260928.
"""
from pathlib import Path
import csv, hashlib, json, math, sys
import numpy as np
from xlsx_values import sheets, records

ROOT = Path(__file__).resolve().parents[1]
QREFS = ['nominal', 'first_comparable', 'early_median5_comparable']
RF, GB = 'Random Forest Regressor', 'Gradient Boosting Regressor'

def csvwrite(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    source = ROOT/'project/outputs/xjtu_a8_deterministic_reconstruction_v1/xjtu_a8_cell_metrics.csv'
    with source.open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    lookup = {}
    for row in rows:
        if row['Model'] in [RF, GB] and row['Qref_Definition'] in QREFS:
            key = (row['Cell_ID'], row['Qref_Definition'], row['Model'])
            if key in lookup: raise ValueError('Duplicate cell/reference/model: '+str(key))
            lookup[key] = float(row['MAE_Q_Ah'])
    cells = sorted({key[0] for key in lookup})
    assert len(cells) == 55 and len(lookup) == 55 * 3 * 2
    values = np.array([[[lookup[(cell, q, m)] for m in [RF, GB]] for q in QREFS] for cell in cells])
    if not np.isfinite(values).all(): raise ValueError('Nonfinite cell metrics')
    draws = np.random.default_rng(20260928).integers(0, 55, size=(5000, 55), dtype=np.int64)
    draw_hash = hashlib.sha256(draws.astype('<i8', copy=False).tobytes()).hexdigest()
    assert draw_hash == '2e8a1afa9101de2c4af7927c31f1e112adde5be2922437d03422aa90207e1ded'
    med = np.median(values[draws], axis=1)
    observed = np.median(values, axis=0)
    s6 = []
    for i, q in enumerate(QREFS):
        differences = med[:, i, 1] - med[:, i, 0]
        lo, hi = np.quantile(differences, [0.025, 0.975], method='linear')
        s6.append({'Qref_Definition': q, 'Observed_RF_Median_MAE_Q_Ah': observed[i, 0],
                   'Observed_GB_Median_MAE_Q_Ah': observed[i, 1],
                   'Observed_Difference_Of_Medians': observed[i, 1] - observed[i, 0],
                   'Bootstrap_Median_Difference_Of_Medians': float(np.median(differences)),
                   'Bootstrap_95CI_Lower': float(lo), 'Bootstrap_95CI_Upper': float(hi),
                   'CI_Includes_Zero': 'YES' if lo <= 0 <= hi else 'NO',
                   'RF_Lower_Median_Count': int(np.sum(differences > 0)),
                   'RF_Lower_Median_Frequency': float(np.mean(differences > 0)),
                   'GB_Lower_Median_Count': int(np.sum(differences < 0)),
                   'GB_Lower_Median_Frequency': float(np.mean(differences < 0)),
                   'Tie_Count': int(np.sum(differences == 0)), 'Tie_Frequency': float(np.mean(differences == 0)),
                   'Bootstrap_Replicates': 5000, 'Bootstrap_Seed': 20260928, 'Cell_Count': 55,
                   'Bootstrap_Draw_SHA256': draw_hash})
    nominal_rf_lower = med[:, 0, 0] < med[:, 0, 1]
    s7 = []
    for i, q in enumerate(QREFS[1:], 1):
        measured_gb_lower = med[:, i, 1] < med[:, i, 0]
        s7.append({'Measured_Qref_Definition': q,
                   'Observed_Nominal_Difference_Of_Medians': observed[0, 1] - observed[0, 0],
                   'Observed_Measured_Difference_Of_Medians': observed[i, 1] - observed[i, 0],
                   'Nominal_RF_Lower_Count': int(nominal_rf_lower.sum()),
                   'Nominal_RF_Lower_Frequency': float(nominal_rf_lower.mean()),
                   'Measured_GB_Lower_Count': int(measured_gb_lower.sum()),
                   'Measured_GB_Lower_Frequency': float(measured_gb_lower.mean()),
                   'Joint_Rankflip_Count': int((nominal_rf_lower & measured_gb_lower).sum()),
                   'Observed_Sample_RankFlip_Stability': float((nominal_rf_lower & measured_gb_lower).mean()),
                   'Bootstrap_Replicates': 5000, 'Bootstrap_Seed': 20260928, 'Cell_Count': 55,
                   'Bootstrap_Draw_SHA256': draw_hash})
    wb = sheets(ROOT/'supplementary/original/Supplementary_Tables_S1-S9.xlsx')
    comparisons, differences = 0, []
    for name, generated, key in [('S6_A8_Corrected_CI', s6, 'Qref_Definition'), ('S7_A8_RankFlip', s7, 'Measured_Qref_Definition')]:
        expected = {r[key]: r for r in records(wb[name])}
        assert len(expected) == len(generated)
        for row in generated:
            for k, v in row.items():
                actual = expected[row[key]][k]
                comparisons += 1
                if isinstance(v, (float, int)):
                    ok = math.isclose(float(actual), v, rel_tol=1e-10, abs_tol=1e-12)
                else: ok = actual == str(v)
                if not ok: differences.append({'table': name, 'qref': row[key], 'column': k, 'workbook': actual, 'reproduced': v})
    dest = ROOT/'reruns'; dest.mkdir(exist_ok=True)
    csvwrite(dest/'S6_reproduced.csv', s6); csvwrite(dest/'S7_reproduced.csv', s7)
    report = {'status': 'PASS' if not differences else 'FAIL', 'comparisons': comparisons,
              'differences': differences, 'numpy': np.__version__, 'python': sys.version.split()[0],
              'bootstrap_draw_sha256': draw_hash, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'statistic': 'median(GB cell MAE_Q) - median(RF cell MAE_Q)', 'cell_order': 'lexicographic Cell_ID'}
    (dest/'S6_S7_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not differences else 1

if __name__ == '__main__': raise SystemExit(main())
