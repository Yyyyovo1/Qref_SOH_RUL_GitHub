import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "file:///LOCAL_HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const reportPath = path.resolve(
  process.argv[2] ??
    "outputs/xjtu_a8_artifact_recovery_v1/xjtu_a8_artifact_recovery_report.csv",
);
const csvText = await fs.readFile(reportPath, "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "RecoveryAudit" });
const inspection = await workbook.inspect({
  kind: "region",
  sheetId: "RecoveryAudit",
  range: "A1:E20",
  maxChars: 12000,
  tableMaxRows: 20,
  tableMaxCols: 5,
  tableMaxCellChars: 260,
});
const serialized = inspection.ndjson ?? String(inspection);
for (const required of [
  "Artifact_Type",
  "Candidate_Path",
  "Found",
  "Usable",
  "Reason",
  "A8_AGGREGATE_SUMMARY",
  "ROBUSTNESS_PIPELINE_PERSISTENCE_AUDIT",
]) {
  if (!serialized.includes(required)) {
    throw new Error(`ARTIFACT_TOOL_REPORT_CONTENT_MISMATCH:${required}`);
  }
}
console.log("ARTIFACT_TOOL_CSV_VALIDATION_PASS");
console.log(serialized);
