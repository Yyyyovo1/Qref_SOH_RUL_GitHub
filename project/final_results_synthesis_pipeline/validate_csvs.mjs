import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const workspace = process.argv[2];
if (!workspace) throw new Error("Usage: node validate_csvs.mjs <workspace>");
const outDir = path.join(workspace, "outputs", "final_results_synthesis_v1");
const reportPath = path.join(workspace, "final_results_synthesis_pipeline", "artifact_validation_report.json");

const expected = {
  "table_dataset_overview.csv": ["Dataset", "Cell_Count", "Primary_Observation_Count", "Nominal_Qref_Ah", "Available_Qref_Definitions", "Initial_Test_Available_Cell_Count", "Primary_Evaluation_Context", "Observation_Density", "EOL_Censoring_Context", "Dataset_Stopping_Rule_Context", "Primary_Role_In_Study"],
  "table_qref_effect_summary.csv": ["Dataset", "Qref_Definition", "Comparison_Scope", "Cell_Count", "Qref_Available_Count", "Delta_Qref_vs_Nominal_Ah_Min", "Delta_Qref_vs_Nominal_Ah_Median", "Delta_Qref_vs_Nominal_Ah_Max", "Delta_Qref_vs_Nominal_Pct_Min", "Delta_Qref_vs_Nominal_Pct_Median", "Delta_Qref_vs_Nominal_Pct_Max", "Qref_Direction", "80pct_Threshold_Shift_Ah_Min", "80pct_Threshold_Shift_Ah_Median", "80pct_Threshold_Shift_Ah_Max"],
  "table_eol_80pct_summary.csv": ["Dataset", "Qref_Definition", "Cell_Count", "EXACT_Count", "INTERVAL_Count", "RIGHT_Count", "Point_RUL_Available_Count", "Point_RUL_Available_Fraction", "Comparison_vs_Nominal", "Status_Changed_Count", "RUL_Availability_Changed_Count", "Exact_Delta_Valid_Count", "Lower_Bound_Valid_Count", "Exact_Delta_EOL_Cycles_Median", "Minimum_Observed_Earlier_By_Cycles_Median"],
  "table_threshold_robustness.csv": ["Dataset", "Threshold_Fraction", "Threshold_Percent", "Qref_Definition", "Comparison_Scope", "Cell_Count", "Qref_Available_Count", "EXACT_Count", "INTERVAL_Count", "RIGHT_Count", "Point_RUL_Available_Count", "Point_RUL_Available_Fraction", "Relative_Comparison", "Qref_Induced_Label_Effect_Cell_Count", "EOL_Status_Changed_Cell_Count", "RUL_Point_Availability_Changed_Cell_Count", "Exact_EOL_Shift_Cell_Count", "Both_Right_Censored_Cell_Count", "Observation_Window_Limited", "Cross_Threshold_Classification", "Dataset_Stopping_Rule_Context"],
  "table_soh_ml_primary.csv": ["Dataset", "Qref_Definition", "Model", "Cell_Count", "Observation_Count", "Median_MAE_Q_Ah", "IQR_MAE_Q_Ah", "Median_RMSE_Q_Ah", "Median_R2_Q", "Median_MAE_SOH_percentage_points", "Median_RMSE_SOH_fraction", "Median_R2_SOH", "Rank_Position_Under_Median_Cell_MAE_Q", "Primary_Ranking_Metric", "Ranking_Terminology"],
  "table_soh_ml_robustness.csv": ["Experiment", "Feature_Set", "Dataset", "Qref_Definition", "Model", "Cell_Count", "Observation_Count", "Median_MAE_Q_Ah", "Rank_Position_Under_Median_Cell_MAE_Q", "Ranking_Order", "Ranking_Changed_vs_Primary", "A8_Corrected_RF_GB_Observed_Difference_Of_Medians", "A8_Corrected_RF_GB_95CI_Lower", "A8_Corrected_RF_GB_95CI_Upper", "A8_Corrected_Bootstrap_Interpretation", "Ranking_Metric"],
  "supplementary_plan.csv": ["Supplement_ID", "Proposed_Item", "Recommendation", "Rationale"],
  "discussion_evidence_map.csv": ["Discussion_Point", "Supporting_Dataset", "Supporting_Result", "Strength", "Important_Limitation", "Recommended_Wording"],
  "claim_strength_audit.csv": ["Claim_ID", "Candidate_Claim", "Classification", "Evidence_Basis", "Important_Limitation"],
  "final_results_consistency_tests.csv": ["test_id", "status", "observed", "expected", "source_files", "notes"],
};

const results = [];
for (const [name, headers] of Object.entries(expected)) {
  const csvText = await fs.readFile(path.join(outDir, name), "utf8");
  const wb = await Workbook.fromCSV(csvText, { sheetName: "Data" });
  const sheet = wb.worksheets.getItem("Data");
  const used = sheet.getUsedRange(true);
  if (!used) throw new Error(`${name}: empty CSV`);
  const values = used.values;
  const actualHeaders = values[0].map((v) => String(v ?? ""));
  if (JSON.stringify(actualHeaders) !== JSON.stringify(headers)) {
    throw new Error(`${name}: header mismatch\nactual=${JSON.stringify(actualHeaders)}\nexpected=${JSON.stringify(headers)}`);
  }
  if (values.length < 2) throw new Error(`${name}: no data rows`);
  const errors = await wb.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 50 },
    summary: `formula-like error scan for ${name}`,
  });
  if (errors.ndjson && /#REF!|#DIV\/0!|#VALUE!|#NAME\?|#NUM!|#NULL!|#SPILL!|#CALC!/.test(errors.ndjson)) {
    throw new Error(`${name}: error token detected`);
  }
  results.push({ file: name, rows: values.length - 1, columns: actualHeaders.length, status: "PASS" });
}

const report = {
  status: "PASS",
  validated_csv_count: results.length,
  validator: "@oai/artifact-tool Workbook.fromCSV + header/data/error inspection",
  files: results,
};
await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + "\n", "utf8");
console.log(JSON.stringify(report));
