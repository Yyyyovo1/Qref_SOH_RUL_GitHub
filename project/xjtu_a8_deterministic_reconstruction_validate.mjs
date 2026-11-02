import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "file:///LOCAL_HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const outputDirectory = path.resolve(
  process.argv[2] ?? "outputs/xjtu_a8_deterministic_reconstruction_v1",
);

async function inspectCsv(fileName, sheetName, range, sampleLarge = false) {
  const fullPath = path.join(outputDirectory, fileName);
  const csvText = await fs.readFile(fullPath, "utf8");
  let importText = csvText;
  if (sampleLarge) {
    const lines = csvText.trimEnd().split(/\r?\n/);
    importText = [lines[0], ...lines.slice(1, 7), ...lines.slice(-6)].join("\n");
  }
  const workbook = await Workbook.fromCSV(importText, { sheetName });
  const inspection = await workbook.inspect({
    kind: "region",
    sheetId: sheetName,
    range,
    maxChars: 18000,
    tableMaxRows: 50,
    tableMaxCols: 30,
    tableMaxCellChars: 260,
  });
  const serialized = inspection.ndjson ?? String(inspection);
  if (!serialized || !serialized.includes(sheetName)) {
    throw new Error(`ARTIFACT_TOOL_INSPECTION_EMPTY:${fileName}`);
  }
  return serialized;
}

const checks = [
  [
    "xjtu_a8_heldout_predictions.csv",
    "Predictions",
    "A1:R13",
    true,
    ["Cell_ID", "Observed_SOH_Fraction", "Predicted_Capacity_Ah", "Absolute_Error_Q_Ah"],
  ],
  [
    "xjtu_a8_cell_metrics.csv",
    "CellMetrics",
    "A1:R12",
    false,
    ["MAE_Q_Ah", "RMSE_Q_Ah", "R2_Q", "Aggregation_Unit"],
  ],
  [
    "xjtu_a8_reproduction_audit.csv",
    "Reproduction",
    "A1:Q10",
    false,
    ["Absolute_Reproduction_Error_Ah", "Median_Reproduction_Status", "PASS"],
  ],
  [
    "xjtu_a8_corrected_difference_of_medians_ci.csv",
    "CorrectedCI",
    "A1:AB4",
    false,
    ["Observed_Difference_Of_Medians", "Bootstrap_95CI_Lower", "Primary_Statistic"],
  ],
  [
    "xjtu_a8_corrected_rankflip_frequency.csv",
    "RankFlip",
    "A1:V3",
    false,
    ["Observed_Sample_RankFlip_Stability", "Same_Draws_Across_Qrefs"],
  ],
  [
    "xjtu_a8_corrected_rank_frequency.csv",
    "RankFrequency",
    "A1:T10",
    false,
    ["Bootstrap_Rank1_Frequency", "median held-out-cell MAE_Q_Ah"],
  ],
  [
    "xjtu_a8_reconstruction_manifest.csv",
    "Manifest",
    "A1:F50",
    false,
    ["INPUT_FILE", "BOOTSTRAP_DRAW_SHA256", "SOFTWARE_VERSION"],
  ],
  [
    "xjtu_a8_reconstruction_regression_tests.csv",
    "Regression",
    "A1:F33",
    false,
    ["ALL_NINE_FROZEN_MEDIANS_REPRODUCED", "BOOTSTRAP_REPLICATES_5000", "PASS"],
  ],
];

for (const [fileName, sheetName, range, sampleLarge, required] of checks) {
  const serialized = await inspectCsv(fileName, sheetName, range, sampleLarge);
  for (const token of required) {
    if (!serialized.includes(token)) {
      throw new Error(`ARTIFACT_TOOL_CONTENT_MISMATCH:${fileName}:${token}`);
    }
  }
  console.log(`ARTIFACT_TOOL_CSV_PASS:${fileName}`);
}

console.log("ARTIFACT_TOOL_ALL_XJTU_A8_OUTPUTS_PASS");
