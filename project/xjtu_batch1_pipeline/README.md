# XJTU Batch-1 QC + Qref Sensitivity Pilot Pipeline

本项目把 XJTU Battery Dataset Batch-1 的原始 MAT 数据处理为可追溯的循环级 QC、四种参考容量（Qref）、SOH、EOL/右删失和 RUL 敏感性结果。程序不修改原始 MAT，不平滑容量，不插值循环，不删除或重编号原始循环，也不训练预测模型。

## 输入与文件角色

默认输入目录是项目根目录下的 `Batch-1/`。分析输入只能是：

- `2C_battery-1.mat` 至 `2C_battery-8.mat`：`CANONICAL`，纳入分析。
- `2C_battery-1(1).mat` 至 `2C_battery-8(1).mat`：仅在 SHA-256 与对应 canonical 文件完全一致时登记为 `DUPLICATE_CONFIRMED`，不进入任何循环级计算或统计。

如果 canonical ID 不是完整的 1–8、存在未识别 MAT 文件，或任一 duplicate 与 canonical 的 SHA-256 不一致，程序立即停止。原始文件在处理前后都会重新计算 SHA-256，以验证未被修改。

## 环境与运行

在 PowerShell 中进入本目录后运行：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_pipeline.py
```

程序先只处理 `2C_battery-1.mat` 并运行回归测试；任何回归测试失败都会停止，不会继续处理 battery 2–8。正式输出目录已存在时程序拒绝覆盖，需使用新的输出版本名后再运行。

人工复核结论确认后，可在不修改自动 QC 输出的情况下生成最终版：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_finalization.py
```

最终化程序只读取 `outputs/xjtu_batch1_pilot_v1/`，把人工决定追加为独立字段，并输出到 `outputs/xjtu_batch1_pilot_final_v1/`。若最终目录已存在、自动输出或原始 MAT 哈希发生变化，程序会停止。

## 配置

所有实验协议、QC、容量一致性、容量跳变、EOL 与回归测试阈值集中保存在 `xjtu_batch1_config.json`。容量主值来自 `summary.discharge_capacity_Ah`；逐循环原始电流、电压和容量曲线用于验证充放电协议与容量一致性。

容量跳变只与同一电池、同一 protocol class 下最近的有效可比较前序循环比较。容量跳变单独只能产生 `CHECK`，不能产生 `INVALID_CANDIDATE`。

## 输出

默认输出目录：`outputs/xjtu_batch1_pilot_v1/`

- `xjtu_file_manifest.csv`：16 个 MAT 的角色、SHA-256、schema 和纳入状态。
- `xjtu_cycle_capacity_qc.csv`：全部 canonical 原始循环；无删除、无重编号。
- `xjtu_qc_flagged_cycles.csv`：全部 `CHECK` 和 `INVALID_CANDIDATE`。
- `xjtu_qref_summary.csv`：nominal、initial test、first standard、early standard median-5 及来源循环。
- `xjtu_soh_labels_long.csv`：可分析标准老化循环的未截断 SOH 标签。
- `xjtu_eol_sensitivity.csv`：各 Qref 及固定 1.600 Ah 阈值的 EOL/右删失结果。
- `xjtu_eol_pairwise_shift.csv`：Qref 两两 EOL 位移与删失状态变化。
- `xjtu_rul_sensitivity.csv`：仅 observed EOL 生成数值 RUL；右删失 EOL 的 RUL 保持空值。
- `xjtu_pilot_summary.csv`：每只电池的 QC、Qref 和 EOL 摘要。
- `processing_log.txt`：环境、输入、配置、分阶段计数、回归测试和最终状态。
- `figures/`：8 张容量/QC 图、Qref 对比图和 EOL 敏感性总览图。

## 核心索引和纳入规则

- `Raw_Cycle_Index` 永远对应 MAT 原始循环号。
- `Aging_Cycle_Index = Raw_Cycle_Index - 1`；Cycle 1 是 initial diagnostic cycle，不与标准老化循环混合计算 Qref。
- `PASS → Primary_Include_Status=YES`。
- `CHECK → REVIEW`；若唯一原因是完整协议下的 `CAPACITY_JUMP`，仅 `Provisional_Include=YES`，QC 仍为 `CHECK`。
- `INVALID_CANDIDATE → NO`，但该循环仍完整保留在 QC 表。
- `RIGHT_CENSORED` 不以最后观测循环冒充 EOL，相关 RUL 保持 `NA`。

## 可复现性检查

流水线结束前会重新读取所有输出 CSV，验证唯一键、Cell ID、循环编号、Qref 来源、INVALID 排除、右删失 RUL 空值及原始文件哈希。只有全部检查通过，日志末尾才会写入 `ALL REGRESSION TESTS PASSED` 和 `PIPELINE COMPLETED SUCCESSFULLY`。
