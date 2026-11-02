import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "file:///LOCAL_HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const outputDirectory = path.resolve(
  process.argv[2] ?? "outputs/soh_feature_extraction_v1",
);

async function inspectCsv(fileName, sheetName, range) {
  const csvText = await fs.readFile(path.join(outputDirectory, fileName), "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName });
  const inspection = await workbook.inspect({
    kind: "region",
    sheetId: sheetName,
    range,
    maxChars: 6000,
    tableMaxRows: 10,
    tableMaxCols: 18,
    tableMaxCellChars: 140,
  });
  const serialized = inspection.ndjson ?? String(inspection);
  if (!serialized || !serialized.includes(sheetName)) {
    throw new Error(`ARTIFACT_TOOL_INSPECTION_EMPTY:${fileName}`);
  }
  return serialized;
}

const regression = await inspectCsv(
  "feature_extraction_regression_tests.csv",
  "Regression",
  "A1:J10",
);
const missingness = await inspectCsv(
  "feature_missingness_summary.csv",
  "Missingness",
  "A1:R8",
);

if (!regression.includes("XJTU_EXPECTED_OBSERVATION_KEYS") || !regression.includes("PASS")) {
  throw new Error("ARTIFACT_TOOL_REGRESSION_CONTENT_MISMATCH");
}
if (!missingness.includes("Feature_Name") || !missingness.includes("Missing_Rate")) {
  throw new Error("ARTIFACT_TOOL_MISSINGNESS_CONTENT_MISMATCH");
}

console.log("ARTIFACT_TOOL_CSV_VALIDATION_PASS");
console.log(regression);
console.log(missingness);
