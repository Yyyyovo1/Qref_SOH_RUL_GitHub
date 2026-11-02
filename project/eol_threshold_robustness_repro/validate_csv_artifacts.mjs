import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const outputDir = process.argv[2];
if (!outputDir) {
  throw new Error("Usage: node validate_csv_artifacts.mjs <output-dir>");
}

const files = [
  "eol_threshold_cell_level.csv",
  "eol_threshold_status_summary.csv",
  "eol_threshold_transition_summary.csv",
  "eol_threshold_rul_availability.csv",
  "eol_threshold_effect_bounds.csv",
  "eol_threshold_qref_direction_audit.csv",
  "eol_threshold_hust_stopping_rule_audit.csv",
  "eol_threshold_cross_threshold_summary.csv",
  "eol_threshold_regression_tests.csv",
];

const results = [];
for (const file of files) {
  const filePath = path.join(outputDir, file);
  const csvText = await fs.readFile(filePath, "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Data" });
  const inspection = await workbook.inspect({
    kind: "table",
    range: "Data!A1:AZ8",
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: 52,
    maxChars: 8000,
  });
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 50 },
    summary: `formula error scan for ${file}`,
    maxChars: 3000,
  });
  results.push({
    file,
    bytes: Buffer.byteLength(csvText, "utf8"),
    tableInspectionChars: inspection.ndjson.length,
    formulaErrorInspection: errors.ndjson,
  });
}

console.log(JSON.stringify({ validated: files.length, results }, null, 2));
