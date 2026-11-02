import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const workspace = path.resolve(process.argv[2] ?? process.cwd());
const outputDir = path.join(
  workspace,
  "outputs",
  "xjtu_a8_rankflip_verification_v1",
);
const expected = {
  "xjtu_a8_rf_gb_bootstrap_ci.csv": {
    rows: 3,
    cols: 21,
    keys: ["Qref_Definition"],
  },
  "xjtu_a8_rank_frequency.csv": {
    rows: 9,
    cols: 18,
    keys: ["Qref_Definition", "Model"],
  },
  "xjtu_a8_qref_rankflip_bootstrap.csv": {
    rows: 2,
    cols: 22,
    keys: ["Measured_Qref_Definition"],
  },
  "xjtu_a8_rankflip_regression_tests.csv": {
    rows: 11,
    cols: 6,
    keys: ["Test_ID"],
  },
};

function columnName(number) {
  let result = "";
  while (number > 0) {
    number -= 1;
    result = String.fromCharCode(65 + (number % 26)) + result;
    number = Math.floor(number / 26);
  }
  return result;
}

const results = [];
for (const [file, spec] of Object.entries(expected)) {
  const csv = await fs.readFile(path.join(outputDir, file), "utf8");
  const workbook = await Workbook.fromCSV(csv, { sheetName: "Data" });
  const sheet = workbook.worksheets.getItem("Data");
  const address = `A1:${columnName(spec.cols)}${spec.rows + 1}`;
  const values = sheet.getRange(address).values;
  const headers = values[0].map((value) =>
    String(value ?? "").replace(/^\uFEFF/, ""),
  );
  const region = await workbook.inspect({
    kind: "region",
    sheetId: "Data",
    range: address,
    maxChars: 2500,
  });
  const record = region.records[0];
  const keyIndexes = spec.keys.map((key) => headers.indexOf(key));
  const keyValues = values.slice(1).map((row) =>
    keyIndexes.map((index) => String(row[index] ?? "")),
  );
  const keyStrings = keyValues.map((row) => JSON.stringify(row));
  const regressionStatusIndex = headers.indexOf("Status");
  const regressionPass =
    regressionStatusIndex < 0 ||
    values.slice(1).every((row) => row[regressionStatusIndex] === "PASS");
  const valid =
    values.length === spec.rows + 1 &&
    values.every((row) => row.length === spec.cols) &&
    headers.every((header) => header.trim() !== "") &&
    keyIndexes.every((index) => index >= 0) &&
    keyValues.every((row) => row.every((value) => value !== "")) &&
    new Set(keyStrings).size === keyStrings.length &&
    record.rows === spec.rows + 1 &&
    record.cols === spec.cols &&
    regressionPass;
  if (!valid) {
    throw new Error(`Artifact validation failed: ${file}`);
  }
  results.push({
    file,
    rows: spec.rows,
    columns: spec.cols,
    firstHeader: headers[0],
    lastHeader: headers.at(-1),
    valid,
  });
}

console.log(JSON.stringify(results, null, 2));
console.log("ARTIFACT_TOOL_VALIDATION=PASS");
