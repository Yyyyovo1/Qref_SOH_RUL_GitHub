# CALCE CS2 单循环放电容量与协议完整性 QC

本项目对 University of Maryland CALCE 的 CS2-35、CS2-36、CS2-37 与 CS2-38 寿命测试工作簿进行可重复的只读处理。当前阶段只完成：

1. 单循环放电容量提取；
2. CC、CV 和主放电阶段完整性检查；
3. 文件边界、局部 Cycle 编号、累计容量计数器和重复文件审计；
4. CSV、日志和原始容量曲线输出。

程序不计算 SOH、EOL 或 RUL，不训练模型，不平滑、不插值、不替换容量，也不删除任何 Cycle。

## 固定目录

| Cell | 默认输入 | 默认输出根目录 |
|---|---|---|
| CS2-35 | 工作区 `CS2_35/` | `outputs/` |
| CS2-36 | 工作区 `CS2_36/` | `outputs/CS2_36/` |
| CS2-37 | 工作区 `CS2_37/` | `outputs/CS2_37/` |
| CS2-38 | 工作区 `CS2_38/` | `outputs/CS2_38/` |

每次运行建立新的 UTC 时间戳子目录，不覆盖已有输出。程序以只读模式打开源工作簿，并在结束前重新计算全部源文件 SHA-256；任何源文件哈希变化都会使运行失败。

## 安装与运行

建议使用 Python 3.12：

```powershell
cd calce_cs2_35_pipeline
python -m pip install -r requirements.txt
python run_pipeline.py --cell-id CS2-37
```

处理 CS2-38：

```powershell
python run_pipeline.py --cell-id CS2-38
```

CS2-35 仍可重复运行：

```powershell
python run_pipeline.py --cell-id CS2-35
```

也可以显式覆盖输入或输出目录：

```powershell
python run_pipeline.py --cell-id CS2-36 --input-dir ..\CS2_36 --output-root .\outputs\CS2_36
```

为修订版 QC 建立独立的 `V2/run_v2_*` 输出，不覆盖旧结果：

```powershell
python run_pipeline.py --cell-id CS2-35 --output-root .\outputs\V2\CS2_35 --run-prefix run_v2
python run_pipeline.py --cell-id CS2-36 --output-root .\outputs\V2\CS2_36 --run-prefix run_v2
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

## 批处理前强制基准

- CS2-35：`CS2_35_1_10_11(1).xlsx` Cycle 2，人工核准值 `0.8699526 Ah`，容差 `1e-6 Ah`。
- CS2-36：`CS2_36_10_28_10.xlsx` Cycle 2，Statistics 相邻累计差与 Channel 循环内首尾差独立交叉验证值 `0.9963362809876899 Ah`，容差 `1e-9 Ah`。
- CS2-37：`CS2_37_10_28_10.xlsx` Cycle 2，Statistics 相邻累计差与 Channel 循环内首尾差独立交叉验证值 `0.9691913622921109 Ah`，容差 `1e-9 Ah`。
- CS2-38：`CS2_38_10_28_10.xlsx` Cycle 2，Statistics 相邻累计差与 Channel 循环内首尾差独立交叉验证值 `0.96857087639073 Ah`，容差 `1e-9 Ah`。

两个来源都必须通过对应基准，否则批处理立即停止。CS2-36、CS2-37 与 CS2-38 的基准是程序化交叉验证基准，不宣称是外部人工给定值。

## 容量提取

对于含 Statistics 表且具有可靠前值的 Cycle：

```text
Q_k = Statistics.D_end,k - Statistics.D_end,k-1
```

文件内第一 Cycle 使用该 Cycle 在 Channel 中的起始累计值作为基线。

当文件没有 Statistics 表，或末尾残缺 Cycle 未出现在 Statistics 中时：

```text
Q_k = Channel.D_end,k - Channel.D_start,k
```

程序保留 Statistics 与 Channel 两套容量及其差值，并在 `Capacity_Source` 中记录最终来源。已确认 CS2-35、CS2-36、CS2-37 与 CS2-38 的每个原始文件都从零重新开始 Channel 放电累计计数器。计数器下降只触发标记，不会进行分段重算、平滑或替换。

## 阶段识别与阈值

`Step_Index` 只用于连续分段和保留辅助证据。CC、CV 与主放电是否存在及是否完整，同时依据电流、电压、`Step_Time`、充放电容量增量和阶段顺序。

所有阈值集中在 `src/calce_qc/config.py`：

| 检查 | 默认阈值 |
|---|---|
| CC 电流 | 0.40–0.70 A |
| CC 完整性 | 至少 5 条、300 s、充电增量 0.02 Ah、末端电压至少 4.15 V |
| CV 电压 | 4.15–4.25 V |
| CV 完整性 | 至少 5 条、300 s、充电增量 0.005 Ah、末端电流不高于 0.08 A |
| 主放电候选电流 | 不高于 -0.80 A |
| 期望主放电平均电流 | -1.25 至 -0.95 A |
| 放电完整性 | 至少 5 条、300 s、容量增量 0.05 Ah |
| 正常放电截止电压检查区间 | 2.55–2.85 V |
| Statistics/Channel 容量差 | 超过 0.001 Ah 时 CHECK |
| 容量跳变 | 相对最近一个可比较有效 Cycle，同时达到 0.12 Ah 和 15% 时 CHECK |

Capacity Jump 不再固定使用紧邻的前一个观测 Cycle。程序沿 Global Cycle 向前寻找最近一个同时满足以下条件的参考 Cycle：主序列文件、边界已确认、容量大于 0、CV 与放电无不完整标记、容量不是非正值、主放电协议完整，且该参考 Cycle 本身不是 `INVALID_CANDIDATE`。不合格的相邻 Cycle 会被跳过。

容量跳变永远只产生 CHECK，不会删除、替换或直接判定 Cycle 无效。输出同时记录 `Jump_Reference_Global_Cycle`、`Jump_Reference_Capacity_Ah` 和 `Jump_Delta_Ah`。若此前没有合格参考，则 `Jump_Reference_Status=NO_VALID_JUMP_REFERENCE`，不产生 Capacity Jump，并在 `Jump_Reference_Reason` 中说明原因。

## CS2-36 边界规则

结构扫描确认：

- 26 个工作簿按 Channel 内部时间戳可唯一排序；
- 每个工作簿的 `Cycle_Index` 从 1 重新开始，文件内连续；
- 每个工作簿的 Channel `Discharge_Capacity` 从 0 重新开始；
- 未发现重复文件；
- `CS2_36_10_28_10.xlsx` 起提供 Statistics 表；部分文件的最后一个残缺 Cycle 不在 Statistics 中。

以下 6 个尾部 Cycle 不自动拼接，保留原值并标记 `NEEDS_BOUNDARY_CHECK`：

- `CS2_36_9_7_10.xlsx` Cycle 44；
- `CS2_36_10_04_10.xlsx` Cycle 8；
- `CS2_36_10_05_10.xlsx` Cycle 9；
- `CS2_36_11_01_10.xlsx` Cycle 17；
- `CS2_36_11_24_10.xlsx` Cycle 17；
- `CS2_36_12_23_10.xlsx` Cycle 28。

Global Cycle 只表示按已确认文件时间顺序排列的原始 Cycle 记录序列，不表示程序把跨文件残缺段拼接成了一个完整 Cycle。

## CS2-37 边界规则

结构扫描确认：

- 27 个工作簿按 Channel 内部时间戳可唯一排序；
- 每个工作簿的 `Cycle_Index` 从 1 重新开始，文件内连续；
- 每个工作簿的 Channel `Discharge_Capacity` 从 0 重新开始；
- 未发现重复文件；
- `CS2_37_10_28_10.xlsx` 起提供 Statistics 表；部分文件的最后一个残缺 Cycle 不在 Statistics 中。

以下 7 个尾部 Cycle 不自动拼接，保留原值并标记 `NEEDS_BOUNDARY_CHECK`：

- `CS2_37_9_7_10.xlsx` Cycle 45；
- `CS2_37_10_04_10.xlsx` Cycle 25；
- `CS2_37_10_05_10.xlsx` Cycle 9；
- `CS2_37_11_01_10.xlsx` Cycle 17；
- `CS2_37_11_24_10.xlsx` Cycle 17；
- `CS2_37_12_23_10.xlsx` Cycle 28；
- `CS2_37_1_28_11.xlsx` Cycle 49。

Global Cycle 仍只表示按已确认时间顺序排列的原始 Cycle 记录，不表示跨文件拼接。

## CS2-38 边界规则

结构扫描确认：

- 28 个工作簿按 Channel 内部时间戳排序；
- 每个工作簿的 `Cycle_Index` 从 1 重新开始，文件内连续；
- 每个工作簿的 Channel `Discharge_Capacity` 从 0 重新开始；
- `CS2_38_2_10_11.xlsx` 与 `CS2_38_2_4_11.xlsx` 的 Channel 和 Statistics 值逐值相同，前者保留为重复候选但不分配 Global Cycle；
- `CS2_38_10_28_10.xlsx` 起提供 Statistics 表；部分文件的最后一个残缺 Cycle 不在 Statistics 中。

以下 7 个尾部 Cycle 不自动拼接，保留原值并标记 `NEEDS_BOUNDARY_CHECK`：

- `CS2_38_9_7_10.xlsx` Cycle 43；
- `CS2_38_10_04_10.xlsx` Cycle 25；
- `CS2_38_10_05_10.xlsx` Cycle 9；
- `CS2_38_11_01_10.xlsx` Cycle 17；
- `CS2_38_11_24_10.xlsx` Cycle 17；
- `CS2_38_12_23_10.xlsx` Cycle 26；
- `CS2_38_1_28_11.xlsx` Cycle 42。

Global Cycle 仍只表示按已确认时间顺序排列的主序列原始 Cycle 记录，不表示跨文件拼接；重复候选文件的记录完整保留在 CSV 中。

## QC 等级

- `PASS`：CC、CV 和主放电协议没有发现明显异常。
- `CHECK`：需要人工复核，例如 CC 过短、容量非正、容量跳变、计数器下降或源间容量不一致。
- `INVALID_CANDIDATE`：有明确协议不完整证据，例如 CV 缺失/近零持续时间、主放电缺失或没有到达合理截止电压。记录仍完整保留。
- `Boundary_Status=NEEDS_BOUNDARY_CHECK`：边界需要人工审计，程序不会自动拼接。

## 每次运行的六个输出

- `file_manifest.csv`：文件顺序、时间、局部 Cycle 范围、累计容量边界、哈希、重复角色和边界状态。
- `cycle_capacity_qc.csv`：全部文件、全部 Cycle 的容量、阶段指标、Jump 参考 Cycle 与 QC 原因。
- `qc_summary.csv`：全记录、主序列和逐文件的 QC 计数。
- `cycle_capacity_curve.png`：主序列原始容量；PASS、CHECK、INVALID_CANDIDATE 和边界待查使用不同标记。
- `qc_flagged_cycles.csv`：所有非 PASS 或边界待查记录。
- `processing_log.txt`：版本、依赖、阈值、基准、逐文件读取、哈希验证和运行摘要。

CSV 使用 UTF-8 with BOM，列顺序固定，数值不经平滑、插值或异常替换。
