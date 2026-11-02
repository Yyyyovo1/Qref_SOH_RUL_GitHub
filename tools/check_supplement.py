"""Compare original supplement S2-S5/S9 with the included frozen processed tables.

Read-only for all original scientific artifacts. Results go to reruns/.
"""
from pathlib import Path
import csv, json, math, re
from xlsx_values import sheets, records

ROOT = Path(__file__).resolve().parents[1]

def load(relative):
    with (ROOT/'project'/relative).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def main():
    wb = sheets(ROOT/'supplementary/original/Supplementary_Tables_S1-S9.xlsx')
    checked, errors = 0, []
    def compare(label, a, b):
        nonlocal checked
        checked += 1
        if str(a) == str(b): return
        if str(a) == '' and str(b).lower() in ['nan', 'none', 'na', '']: return
        try: ok = math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-10)
        except (ValueError, TypeError): ok = False
        if not ok: errors.append({'check': label, 'workbook': a, 'processed': b})

    expected_tables = {f'S{i}' for i in range(1, 10)}
    actual_tables = {re.match(r'S[1-9]', n).group() for n in wb if re.match(r'S[1-9]_', n)}
    if actual_tables != expected_tables: errors.append({'missing_tables': sorted(expected_tables - actual_tables)})
    for name, s in wb.items():
        if s['errors'] or s['uncached_formulas']: errors.append({'sheet': name, 'errors': s['errors'], 'uncached_formulas': s['uncached_formulas']})

    source = load('outputs/eol_threshold_robustness_v1/eol_threshold_status_summary.csv')
    index = {(r['Dataset'], r['Qref_Definition'], r['Threshold_Fraction']): r for r in source}
    for r in records(wb['S2_EOL_Censoring']):
        key = (r['Dataset'], r['Frozen_Artifact_Key'], r['Threshold_Fraction'])
        s = index[key]
        for k, sk in [('Cell_Count', 'Cell_Count'), ('Qref_Available_Count', 'Qref_Available_Count'),
                      ('EXACT_Count', 'EXACT_OBSERVED_Count'), ('INTERVAL_Count', 'INTERVAL_CENSORED_Count'),
                      ('RIGHT_Count', 'RIGHT_CENSORED_Count'), ('Point_RUL_Available_Count', 'Point_RUL_Available_Count'),
                      ('Point_RUL_Available_Fraction', 'Point_RUL_Available_Fraction')]: compare('S2/'+str(key)+'/'+k, r[k], s[sk])

    sources = {d: {r['Cell_ID']: r for r in load(f'outputs/{folder}/{filename}')} for d, folder, filename in [
        ('XJTU', 'xjtu_55_final_v2', 'xjtu_55_qref_summary.csv'), ('HUST', 'hust_77_final_v1', 'hust_77_qref_summary.csv')]}
    xmap = {'Nominal_Qref_Ah': 'Qref_nominal_Ah', 'Initial_Test_Qref_Ah': 'Qref_initial_test_Ah',
            'First_Comparable_Ah': 'Qref_first_comparable_Ah', 'First5_Comparable_Ah': 'Qref_early_median5_comparable_Ah',
            'First_Source_Cycle': 'Qref_First_Comparable_Source_Cycle', 'First5_Source_Cycles': 'Qref_Early_Median5_Comparable_Source_Cycles'}
    hmap = {'Nominal_Qref_Ah': 'Qref_Nominal_Ah', 'Initial_Test_Qref_Ah': 'Qref_Initial_Test_Ah',
            'First_Comparable_Ah': 'Qref_First_Comparable_Ah', 'First5_Comparable_Ah': 'Qref_Early_Median5_Comparable_Ah',
            'First_Source_Cycle': 'Qref_First_Comparable_Source_Cycle', 'First5_Source_Cycles': 'Qref_Early_Median5_Comparable_Source_Cycles'}
    for r in records(wb['S3_PerCell_Qref']):
        s = sources[r['Dataset']][r['Cell_ID']]
        for k, sk in (xmap if r['Dataset'] == 'XJTU' else hmap).items(): compare('S3/'+r['Cell_ID']+'/'+k, r[k], s[sk])

    source = load('outputs/soh_primary_benchmark_v1/soh_primary_summary_metrics.csv')
    idx = {(r['Dataset'], r['Qref_Definition'], r['Model']): r for r in source}
    for r in records(wb['S4_Primary_A9_Metrics']):
        s = idx[(r['Dataset'], r['Frozen_Artifact_Key'], r['Model'])]
        for k in ['Cell_Count', 'Observation_Count', 'Median_MAE_Q_Ah', 'IQR_MAE_Q_Ah', 'Median_RMSE_Q_Ah', 'Median_R2_Q', 'Median_MAE_SOH_percentage_points', 'Median_RMSE_SOH_fraction', 'Median_R2_SOH']:
            compare('S4/'+r['Dataset']+'/'+r['Model']+'/'+k, r[k], s[k])

    source = sum((load('outputs/soh_ml_robustness_v1/'+n) for n in ['soh_robustness_a8_summary.csv', 'soh_robustness_b10_summary.csv', 'soh_robustness_postcycle5_summary.csv']), [])
    idx = {(r['Experiment'], r['Dataset'], r['Qref_Definition'], r['Model']): r for r in source}
    for r in records(wb['S5_Robustness_Full']):
        s = idx[(r['Experiment'], r['Dataset'], r['Qref_Definition'], r['Model'])]
        for k in r:
            if k in s: compare('S5/'+r['Experiment']+'/'+r['Dataset']+'/'+k, r[k], s[k])

    source = load('outputs/reference_availability_audit_v2_corrected/aggregate_reference_availability_v2.csv')
    idx = {(r['Dataset'], r['Reference_Display_Name'], r['Population_Scope']): r for r in source}
    for r in records(wb['S9_Reference_Availability'], header_row=2):
        scope = {'SOH-eligible': 'FROZEN_SOH_ELIGIBLE', 'POSTCYCLE5': 'EXISTING_ROBUSTNESS_POSTCYCLE5'}[r['Population']]
        s = idx[(r['Dataset'], r['Reference'], scope)]
        for k, sk in [('Observations', 'V2_Eligible_Observation_Count'), ('Before', 'V2_Strictly_Before_Reference_Availability_Count'),
                      ('At', 'V2_At_Reference_Availability_Count'), ('After', 'V2_Strictly_After_Reference_Availability_Count')]:
            compare('S9/'+r['Dataset']+'/'+r['Reference']+'/'+k, r[k], s[sk])

    report = {'status': 'PASS' if not errors else 'FAIL', 'compared_values': checked, 'errors': errors,
              'tables_present': sorted(actual_tables), 'hidden_sheets': [n for n, s in wb.items() if s['state'] != 'visible'],
              'coverage': 'S2 endpoint counts; S3 Qref and source cycles; S4 primary numeric metrics; S5 shared frozen columns; S9 aggregate counts. S1/S8 presence only; S6/S7 use reproduce_s6_s7.py.'}
    dest = ROOT/'reruns'; dest.mkdir(exist_ok=True)
    (dest/'supplement_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not errors else 1

if __name__ == '__main__': raise SystemExit(main())
