"""Descriptive tables for Results 3.1-3.3 of the JES manuscript. Reads frozen outputs only; writes to
manuscript/jes_v1/results_data/. Usage (repo root): python manuscript/jes_v1/scripts/results_tables.py"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "manuscript" / "jes_v1" / "results_data"
OUT.mkdir(parents=True, exist_ok=True)
rd = lambda p: pd.read_csv(ROOT / p, encoding="utf-8-sig")

# ---------- A. per-cell reference capacities, all datasets
s3 = rd("supplementary/csv/S3_PerCell_Qref.csv")
a = s3.rename(columns={"Nominal_Qref_Ah": "nominal", "First_Comparable_Ah": "first", "First5_Comparable_Ah": "first5"})[["Dataset", "Cell_ID", "nominal", "first", "first5"]]
mit = rd("project/outputs/mit_124_final_v1/mit_qref_summary.csv")
b = pd.DataFrame({"Dataset": "MIT", "Cell_ID": mit.Cell_ID, "nominal": mit.Qref_Nominal_Ah,
                  "first": mit.Qref_First_Comparable_Ah, "first5": mit.Qref_Early_Median5_Comparable_Ah})
tju = rd("project/outputs/tju_final_v2/tju_qref_summary.csv")
c = pd.DataFrame({"Dataset": "TJU-" + tju.Dataset, "Cell_ID": tju.Cell_ID, "nominal": tju.Qref_Nominal_Ah,
                  "first": tju.Qref_First_Comparable_Ah, "first5": tju.Qref_Early_Median5_Comparable_Ah})
cells = pd.concat([a, b, c], ignore_index=True)
cells.to_csv(OUT / "A_per_cell_qref.csv", index=False)
rows = []
for (ds), g in cells.groupby("Dataset", sort=False):
    for q in ("first", "first5"):
        d = g[q] - g.nominal; pct = 100 * d / g.nominal; thp = 0.80 * g[q] / g.nominal
        rows.append(dict(Dataset=ds, Qref=q, Cells=len(g), Below=int((d < 0).sum()), Above=int((d > 0).sum()),
                         Delta_Ah_Median=d.median(), Delta_Ah_Min=d.min(), Delta_Ah_Max=d.max(),
                         Delta_Pct_Median=pct.median(), Delta_Pct_Min=pct.min(), Delta_Pct_Max=pct.max(),
                         ThetaPrime80_Median=thp.median(), ThetaPrime80_Min=thp.min(), ThetaPrime80_Max=thp.max()))
qsum = pd.DataFrame(rows)
qsum.to_csv(OUT / "A_qref_deviation_summary.csv", index=False)

# ---------- B. endpoint classes at 0.75/0.80/0.85, both rules
lab = rd("project/outputs/protocol_validation_v1/trapD_labels.csv")
lab = lab.rename(columns={"Qref_Definition": "Qref"})
lab["Qref"] = lab.Qref.map({"nominal": "nominal", "first_comparable": "first", "early_median5_comparable": "first5"}).fillna(lab.Qref)
lab = lab[["Dataset", "Qref", "Theta", "Cell_ID", "Strict_Status", "Tolerance_Kind"]]
tl = rd("reruns/tju_label_structure/2_labels_per_cell_setting.csv")
tl = pd.DataFrame({"Dataset": "TJU-" + tl.Dataset, "Qref": tl.Qref, "Theta": tl.Theta, "Cell_ID": tl.Cell_ID,
                   "Strict_Status": tl.Status, "Tolerance_Kind": tl.Tolerance_Kind})
L = pd.concat([lab, tl], ignore_index=True)
L["Theta"] = L.Theta.round(2)
assert set(L.Qref) == {"nominal", "first", "first5"}, set(L.Qref)
rows = []
for (ds, q, t), g in L[L.Theta.isin([0.75, 0.80, 0.85])].groupby(["Dataset", "Qref", "Theta"], sort=False):
    vc = g.Strict_Status.value_counts()
    rows.append(dict(Dataset=ds, Qref=q, Theta=t, Cells=len(g), EXACT=int(vc.get("EXACT", 0)), INTERVAL=int(vc.get("INTERVAL", 0)),
                     RIGHT=int(vc.get("RIGHT", 0)), BASELINE=int(vc.get("BASELINE", 0)),
                     Tolerance_Labeled=int(g.Tolerance_Kind.ne("NO_LABEL").sum()),
                     Tolerance_Clause=int(g.Tolerance_Kind.eq("TOLERANCE").sum())))
cls = pd.DataFrame(rows)
cls.to_csv(OUT / "B_endpoint_classes_075_080_085.csv", index=False)

# ---------- C. labelled share by theta and 90% windows
rows = []
for (ds, q, t), g in L.groupby(["Dataset", "Qref", "Theta"], sort=False):
    rows.append(dict(Dataset=ds, Qref=q, Theta=t, Cells=len(g),
                     Strict_Share=g.Strict_Status.eq("EXACT").mean(), Tolerance_Share=g.Tolerance_Kind.ne("NO_LABEL").mean()))
sh = pd.DataFrame(rows)
sh.to_csv(OUT / "C_labelled_share_by_theta.csv", index=False)
rows = []
for (ds, q), g in sh.groupby(["Dataset", "Qref"], sort=False):
    for rule, col in (("strict", "Strict_Share"), ("tolerance", "Tolerance_Share")):
        pts = g[g[col] >= 0.90].Theta.sort_values().tolist()
        grid = sorted(g.Theta)
        contiguous = bool(pts) and len(pts) == (grid.index(pts[-1]) - grid.index(pts[0]) + 1)
        rows.append(dict(Dataset=ds, Qref=q, Rule=rule, Window_Points=len(pts), Window_Low=pts[0] if pts else np.nan,
                         Window_High=pts[-1] if pts else np.nan, Contiguous=contiguous, Max_Share=g[col].max(),
                         Grid_Low=grid[0], Grid_High=grid[-1]))
win = pd.DataFrame(rows)
win.to_csv(OUT / "C_windows_90pct.csv", index=False)

# cross-check: windows equal the committed follow-up and TJU outputs
fw = rd("project/outputs/followup_analyses_v1/5_observable_windows.csv")
m = win.merge(fw, on=["Dataset", "Qref", "Rule"], suffixes=("", "_f"))
ok = ((m.Window_Low.fillna(-1) == m.Window_Low_f.fillna(-1)) & (m.Window_High.fillna(-1) == m.Window_High_f.fillna(-1))).all()
ts = rd("reruns/tju_label_structure/2_labeled_share_by_theta.csv")
ts["Dataset"] = "TJU-" + ts.Dataset; ts["Theta"] = ts.Theta.round(2)
mm = sh.merge(ts, on=["Dataset", "Qref", "Theta"], suffixes=("", "_t"))
ok2 = len(mm) == 216 and np.allclose(mm.Strict_Share, mm.Strict_Share_t, atol=1e-6) and np.allclose(mm.Tolerance_Share, mm.Tolerance_Share_t, atol=1e-6)
print("windows match followup:", bool(ok), "| TJU shares match:", bool(ok2), len(m), len(mm))
print(qsum.round(4).to_string())
print(cls[cls.Theta == 0.80].to_string())
print(win.to_string())
