# v4 实现计划（阶段 1，未写代码）

- 日期：2026-11-14；依据：`project/v4_core_claim/PROTOCOL_v4.md`（SHA-256 `f14204f9…5ad9b9`，提交 `74f645a`）。
- 本文件不是存档文件，也不是修订文件；等作者对第 7 节逐条答复后，答复写进实现说明（需要的写成 `AMENDMENT_01.md`），再进入阶段 2。
- 阶段 1 中只读取了代码、方案、准备输出和文件元数据，没有运行任何模型，没有计算任何误差或比较。

## 1. 代码布局（全部为新文件）

| 位置 | 内容 |
| --- | --- |
| `project/core_claim_v4/` | v4 新包（不放进 `v4_core_claim/`，以免和存档文件混在一起） |
| `project/run_v4_core_claim.py` | 命令行：`--dev`、`--output-dir`、`--compare RUN_A RUN_B` |
| `project/tests/test_v4_*.py` | 单元测试和端到端测试，只用合成数据 |
| `project/core_claim_v4/IMPLEMENTATION_NOTES.md` | 澄清、开发运行记录（与 v2、v3 同样的写法） |
| `reruns/v4/<运行名>/` | 输出；目录已存在则拒绝写入；先写临时目录再整体改名（沿用 v2） |

包内模块：

| 模块 | 职责 |
| --- | --- |
| `config.py` | 方案中的全部常数：网格、种子、GB 超参数、δ = 10%、置信水平、判定阈值、存档文件的 SHA-256 |
| `prereg.py` | 运行开始时核对 15 个存档文件的 SHA-256 与 `PREREG_RECORD.md` 一致，不一致就停止 |
| `severson.py` | 实验 1 特征：用 importlib 按路径导入存档的 `reruns/v4_prep/severson_features.py`，直接调用其中的 `cell_features`、`dq_features`（公式与存档逐字相同，不复制代码）；检查 S |
| `exp1_labels.py` | MIT 两种端点规则的标签、EOL > 100 标记、有标签比例、共同窗口；实验 1 的检查 1–3 |
| `exp1_models.py` | variance（OLS）、discharge 和 full（elastic net + 内层重复 4 折）；外层冻结 5 折；原划分（C 条件） |
| `exp1_eval.py` | 电芯绝对百分比误差、平均值的配对 bootstrap、更优方法、受支持反转、H1 的输入、RMSE（描述）、排除 `b1c18`（描述） |
| `exp3.py` | v2/v3 流水线加 GB：数据、样本、特征、t₀、方法、评价集、检查 0–5（按数据集）、共同窗口 |
| `exp2.py` | 受支持一致比例、等效性检验（Δ 的 90% 置信区间）、检出能力 |
| `verdicts.py` | H1、H2、H3 |
| `blinding.py` | 盲态保护（见第 4 节） |
| `pipeline.py` | 串联；`dev=True` 在可估计性之后停止 |
| `v2v3_agreement.py` | §7 第 6 步：v2、v3 已有结果的受支持一致比例；只在 H1–H3 判定写定之后单独运行 |

## 2. 方案逐节对应

### §0 披露、§0.4 不挑选子集

- 不需要代码。判定用的方法对写死在 `config.py`：H1 = {variance–discharge, variance–full}；H2 = H1 + {RF–GB}；H3 = {RF–GB}。其余方法对照常计算，标记为"描述"。

### §1 P 与 C

| 条目 | 实现 |
| --- | --- |
| 定义 | `(Q_ref, 规则)`，Q_ref ∈ {nominal, first, first5}，规则 ∈ {strict, tolerance} |
| 定义对 | 同一规则下的 Q_ref 两两组合，沿用 `QREF_PAIRS` |
| P 的划分 | MIT `mit_cell_fold_assignments.csv`；HUST、TJU-NCA 用各自冻结的折 |
| bootstrap | `rul_ranking_pilot.evaluation.draws_for(n)`：5000 次，种子 20260928，电芯按 Cell_ID 字典序；同一数据集内所有设置共用。MIT n = 124，与 v2 的 MIT 抽样完全相同 |
| 受支持的差异 | 95% 线性分位数区间不跨 0（v2 的 `_interval`、`_better`） |
| C 的划分 | 实验 1：Severson 原划分（训练 41、主测试 43、次测试 40；`1_severson_features_mit124.csv` 中的 `Severson_Split`）；实验 3：冻结 5 折 |

### §2 实验 1

**数据与标签（§2.1）**

- MIT 容量与 Q_ref：v2 的 `_mit_inputs`（冻结 `c4b2787`）。
- 严格规则：v1 的 `detect`（密集检测器），`EXACT` 时取 `EOL_Exact_Index`。
- 容差规则：`protocol_validation.endpoint.tolerance_rule`，`Tolerance_Kind ≠ NO_LABEL` 时取 `Tolerance_EOL_Index`（可能是插值得到的非整数循环）。
- 目标变量：y = EOL 循环数，回归目标为 log₁₀ y；只用 EOL > 100 的电芯。
- θ 网格：0.75–0.95，步长 0.01（与准备脚本相同）。P 只用共同窗口内的点；C 只用 0.80。
- 共同窗口：运行时重算（两种定义下"有标签且 EOL > 100"的比例都 ≥ 90%），要求与存档的 `3_mit_common_windows.csv` 完全一致，否则停止。
- P 的电芯集合：对每个 (规则, 定义对, θ)，取两种定义下都有标签且 EOL > 100 的电芯；同时记录被排除的电芯数（§5.5）。

**模型（§2.2）**

- variance：`LinearRegression`，只用 `DeltaQ_Variance`。
- discharge（13 个特征）、full（20 个特征）：
  1. 用外层训练折的电芯计算 z 标准化的均值和标准差（`StandardScaler`，ddof = 0），作用于训练折和测试折；
  2. 内层：对 r = 0…9，`KFold(4, shuffle=True, random_state=20260928 + r)` 在外层训练折电芯上分折，共 40 个划分；
  3. `ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], alphas=100（默认路径，eps = 1e-3）, cv=上述 40 个划分, n_jobs=1)`。λ 路径由外层训练折数据计算（sklearn 默认做法）；40 个划分的 MSE 等权平均，等于"10 次重复的平均 4 折 MSE"；取最小值对应的 (α, λ)，在整个外层训练折上重新拟合；
  4. 预测值 = 10^ŷ。
- 不删除异常值，不排除电芯；排除 `b1c18` 的结果只作描述，单独输出（范围见第 7 节 Q12）。

**指标与比较（§2.3）**

- 电芯误差 APE = |ŷ − y| / y × 100；汇总为平均值；RMSE（单位为循环）只作描述。
- 方法对：variance–discharge、variance–full、discharge–full；方向记为 A − B，A 在前。
- 对每个 (规则, 定义对, θ ∈ 窗口, 方法对)：两种定义下各做一遍外层 5 折预测；D(q) = mean APE(A) − mean APE(B)；用同一套抽样得到 D(q) 的 95% 置信区间；按 v2 §1e 判定更优方法和受支持反转。
- bootstrap 写法沿用 v2：在全部 124 个电芯上抽样，不属于该集合的电芯记为 NaN，每次重抽样取 nanmean，有效电芯少于 5 个记为缺失，缺失比例 > 5% 时不可估计。与 v2 唯一的区别是统计量用平均值而不是中位数。
- C1：variance–full 中 full 更优；C2：variance–discharge 中 discharge 更优。

**C 条件与 θ = 0.80（§2.4）**

- 6 种定义 × 3 种模型：在原训练集中"有标签且 EOL > 100"的电芯上训练（elastic net 的内层交叉验证照上述写法，只在这些训练电芯内进行），报告训练、主测试、次测试上的平均百分比误差和 RMSE（点估计）。
- 某个子集没有有标签的电芯时，记为"无标签"（严格规则下的次测试集）；另列出各子集的有标签电芯数。

### §3 实验 2

| 小节 | 实现 |
| --- | --- |
| §3.1 一致比例（描述） | 对实验 1、实验 3 中 P 条件下"两个设置都可估计"的比较：两边都判定出更优方法、并且是同一个方法的比例；以及在"两边都判定出更优方法"的比较中结论一致的比例。按实验、数据集、方法对、评价集分别报告 |
| §3.2 等效性 | 范围：实验 1 的 variance–discharge、variance–full；实验 3 的 RF–GB，只用 E1，只取窗口内。Δ_b = D_b(q₂) − D_b(q₁)，用同一套抽样；90% 区间取 5% 和 95% 线性分位数；δ = 0.10 × 基准统计量（实验 1：nominal 下 variance 的平均 APE；实验 3：nominal 下 BASE 的 E1 中位误差）。区间 ⊂ (−δ, δ) 为等效；与 [−δ, δ] 不相交为不等效；其余为精度不足 |
| §3.3 检出能力（描述） | SE₁、SE₂ = 两种定义下 D 的 bootstrap 标准差；P = Φ(m/SE₁ − 1.96)·Φ(m/SE₂ − 1.96)；m ∈ {1·SE, 2·SE, 0.10·基准, 0.20·基准}；报告各实验、各数据集的中位数和四分位数 |

实验 3 需要 bootstrap 向量本身（v2 的 `error_pair_ci` 只返回区间）。v4 写一个新函数，内部调用 v2 的 `per_cell_errors`、`_nanmedian_rows`、`_interval`，返回同样的区间加上抽样向量；测试要求它在合成数据上与 v2 的 `error_pair_ci` 逐位相同。

### §4 实验 3

- 数据：HUST、MIT 用 v2 的 `load_all`、`load_t0`、`build_samples`、`base_features`；TJU-NCA 用 v3 的 `load_tju`、`detect_all`、`compute_t0_1`、`base_features`，t₀ = 6/77/110，θ 网格为 0.60–0.95。
- 方法：调用 v2 的 `run_setting`（不改），另写 `gb_fold_predictions`，折循环与 RF 一行一行对应：只用精确电芯的样本，`_fold_preprocess(scale=False)`，每折训练精确电芯 ≥ 5，记录预处理审计。GB 超参数取自 `soh_ml_design_v1` 的方案表，运行时从该 csv 读取并与 `config.py` 中的值核对。方法集合待定（第 7 节 Q6）。
- 评价：v2 的 `Arrays`（另把 GB 的预测加进 `arr.pred`）、`membership`、`per_cell_errors`、`error_pair_ci`、`c_index_pair_ci`、`find_reversals`；方法对列表由 v4 自己生成（包含 GB）。
- 共同窗口：严格规则下精确电芯比例（分母为全部电芯，同 v3 的 `strict_shares`）两种 Q_ref 都 ≥ 90%；运行时重算，要求与存档的 `4_exp3_windows_estimability.csv` 的 `In_Common_Window` 列完全一致。窗口限制只加在比较这一层。
- C 条件：θ = 0.80，冻结 5 折，只报告点估计（范围见第 7 节 Q17）。

### §5 判定

| 条目 | 实现 |
| --- | --- |
| H1 | 只看 variance–discharge、variance–full。成立 = 全部定义对上这两个方法对都没有受支持反转，并且在可估计的"设置 × 结论"中，原结论得到支持的至少占一半；被推翻 = 任何一个方法对在同一 (规则, 定义对) 内 ≥ 2 个不同 θ 点出现受支持反转；其余为无法判定。"设置"的单位待定（Q3） |
| H2 | 按顺序：不等效比例 > 5% 为被推翻；否则精度不足比例 > 50% 为无法检验；否则等效比例 ≥ 50% 为成立；其余为无法判定。分母待定（Q5） |
| H3 | 只看 RF–GB，只取通过检查 0–4 和检查 5 的数据集，在窗口内、所有定义对、E1/E2/E3 合计：受支持反转 ≤ 1 为成立，≥ 2 为被推翻；同时报告 20% 档检出概率的中位数，低于 0.5 时注明"检出能力低" |
| 检查 S 未通过 | 实验 1 不参与判定（后果待定，Q9） |
| 结果记录 | `decision.json` 写入每条判定及其计数，并同时保存全部中间表 |

### §6 合理性检查

| 检查 | 实验 1（MIT） | 实验 3（HUST、MIT、TJU-NCA） |
| --- | --- | --- |
| 0 t₀ | 不适用（预测时刻固定为第 100 个循环）；改为核对特征只用第 1–100 个循环（检查 S 的一部分），待确认（Q8） | 沿用 v2（HUST、MIT 对冻结 t₀ 文件）、v3（NCA 对 6/77/110） |
| 1 标签 | 严格规则与 `mit_endpoint_labels.csv` 一致、cycle_life 相差在 ±1 以内（v2 的 `_check_mit_labels`）；容差规则的类型和 EOL 与 `trapD_labels.csv`（`6886053`）逐行一致 | HUST：v1 的 `check_frozen_labels`；MIT：v2 的 `_check_mit_labels`；NCA：v3 的 `_check_labels` |
| 2 物理等价 | 严格和容差两种规则都以"θ·Q_ref 写成 θ'·nominal"重算，标签必须完全一致 | 沿用 v2、v3（包含 TRAJ 等价；是否适用取决于 Q6） |
| 3 单调性 | 严格规则：v1 的 `check_monotonicity`；容差规则：同样逻辑作用于 `Tolerance_EOL_Index` | 沿用 |
| 4 不泄漏 | 见下 | 沿用 v2，含 v3 修订 01 的替代负对照；GB 的审计也纳入 |
| 5 基准偏高 | 没有 BASE，待定（Q8） | 按数据集计算（v3 修订 01）；θ 范围和"最佳方法"的集合待定（Q6、Q7） |
| 6 确定性 | 两次正式运行的输出逐字节相同（v1 的 `compare_runs`） | 同左 |
| 0–4 未通过 | 只排除该数据集或实验，其余继续 | 同左 |

**检查 S（§6）**

1. 重算的 20 个特征与 `1_severson_features_mit124.csv` 比较（比较方式待定，Q1）。
2. `Qdlin`（第 2–100 个循环，1000 × 99）四舍五入到 5 位有效数字后，与 Attia 的数组完全相等（四舍五入的写法待定，Q2）。
3. 124 个电芯的寿命与 Attia 三组寿命逐个相等，划分大小为 41/43/40。

**实验 1 的不泄漏检查（§6）**

- 每个外层折：外层训练电芯与测试电芯不重叠；重算的标准化均值和标准差（只用训练电芯）的 SHA-256 与模型使用的一致；40 个内层划分的全部索引都属于外层训练折；内层划分可由种子重现。
- 负对照：用全部 P 电芯（包含测试折）计算的均值和标准差必须与记录的不同；均值恰好相同时，再比较标准差（沿用 v3 修订 01 的思路）。
- 特征矩阵在 6 种定义下逐字节相同，并且不含 Q_ref 或标签列。

### §7 流程

| 步骤 | 对应 |
| --- | --- |
| 2 写代码并推送 | 阶段 2 |
| 3 开发运行 | `--dev`：检查 0–4、S、实验 1 不泄漏、可估计性（范围待定，Q16） |
| 4 正式运行两次 | `reruns/v4/formal_run_1`、`formal_run_2`，再执行 `--compare` |
| 5 判定和报告 | `decision.json` 和报告 |
| 6 v2、v3 的一致比例 | `v2v3_agreement.py`，在判定之后单独运行，输出到 `reruns/v4/v2v3_agreement/` |

## 3. 复用的代码和输入（只导入，不修改）

| 来源 | 提交 | 复用内容 |
| --- | --- | --- |
| v1 `rul_ranking_pilot` | `8d36e3b`（方案 `3a4ae5c`） | `labels.detect`、`_detection_record`、`check_monotonicity`、`check_frozen_labels`、`load_inputs`、`DatasetInputs`、`PilotError`；`methods._fold_preprocess`、`_sha`、`traj_predict`；`evaluation.draws_for`、`censoring_aware_error`；`features._level`、`_ols_slope`；`pipeline._file_sha`、`_log`、`_write`、`compare_runs`；`config` 中的 `RF_PARAMS`、种子、`MIN_CELLS`、`N_BOOTSTRAP`、`MAX_BOOT_MISSING_FRACTION` |
| v2 `rul_pilot_v2` | 代码 `a318407`（方案 `4cdf0a2`，结果 `99ff103`） | `data._mit_inputs`、`load_all`、`load_t0`、`build_samples`、`base_features`；`methods.run_setting`；`evaluation.Arrays`、`per_cell_errors`、`error_pair_ci`、`c_index_pair_ci`、`membership`、`find_reversals`、`baseline_check`、`_better`、`_interval`、`_nanmedian_rows`、`counts_matrix`；`pipeline._check_mit_labels`、`_physical_equivalence` |
| v3 `rul_pilot_v3_tju` | 代码 `7950cfe`、修订 01 的实现 `f5b92c2`（方案 `332dbf4`，修订 `840bd85`，结果 `19bc78e`） | `data.load_tju`、`detect_all`、`compute_t0_1`、`base_features`；`config.T0`、`THETAS`；`pipeline._check_labels`、`_physical_equivalence`、`strict_shares`。替代负对照写在 v3 的 `run()` 里面，无法导入，v4 按同样逻辑重写一个函数，并用测试核对它在 v3 的情形下给出相同结果 |
| 规范验证 | 代码 `e8c4499`，输出 `6886053` | `protocol_validation.endpoint.tolerance_rule`；`outputs/protocol_validation_v1/trapD_labels.csv`（实验 1 容差规则检查 1 的参照） |
| MIT 冻结 | `c4b2787` | `outputs/mit_124_final_v1/*` |
| TJU 冻结 v2 | `22801aa` | `outputs/tju_final_v2/*`；`reruns/tju_label_structure/2_labels_per_cell_setting.csv`（`332dbf4`） |
| SOH 设计方案 | `47c6f95` | `outputs/soh_ml_design_v1/ml_model_hyperparameter_plan.csv`（GB 一行） |
| v4 准备 | `74f645a`（存档） | `severson_features.py` 中的函数；`1_severson_features_mit124.csv`、`3_mit_common_windows.csv`、`4_exp3_windows_estimability.csv`（作为核对参照） |
| 原始数据 | 不提交 | `raw_data/MIT/*.mat`（特征重算）；`raw_data/attia2021_revisit_severson/`（检查 S） |

环境：`.venv`，Python 3.12.6，numpy 2.5.3、pandas 2.2.3、scipy 1.18.1、scikit-learn 1.7.2、lifelines 0.30.3、h5py 3.14.0，与 v3 正式运行相同。

## 4. 盲态的实现方式

- `pipeline.run(dev=True)` 在可估计性之后返回。所有评价函数（bootstrap、APE 汇总、区间、反转、等效性、检出能力、判定）都放在独立模块中，进入时检查 `blinding.ALLOW_EVALUATION`；开发模式下该开关为假，一旦被调用就抛出异常，而不是悄悄跳过。
- 开发模式的输出用白名单：只允许写检查表、标签和比例、窗口、电芯数、可估计性、审计（哈希值和布尔值）、预测行数。写文件时检查列名，出现 `Pred`、`Error`、`APE`、`RMSE`、`MSE`、`CI`、`Diff`、`Better`、`Reversal`、`Verdict`、`Alpha`、`L1` 等字样就报错。
- 实验 1 的"能拟合"需要真的拟合 elastic net。拟合时会在内存中计算内层交叉验证 MSE（这是选超参数的一部分），开发模式下不写出、不打印 MSE、选中的 (α, λ) 和预测值，只记录"系数和预测是否有限"、收敛警告次数（是否允许这样做见 Q16）。
- 测试只用合成数据；不在真实数据上做打乱标签的运行，除非作者另行要求。
- 日志只写进度和检查结果，不写任何数值型的模型结果。

## 5. 输出文件（`reruns/v4/<运行名>/`）

- **共同：** `prereg_hash_check.csv`、`sanity_checks.csv`、`dataset_checks.csv`、`run_info.json`、`processing_log.txt`；正式运行另有 `decision.json`。
- **实验 1，开发和正式都有：** `exp1_features.csv`、`exp1_checkS_features.csv`、`exp1_checkS_qdlin.csv`、`exp1_checkS_cycle_life.csv`、`exp1_labels.csv`、`exp1_label_shares.csv`、`exp1_common_windows.csv`、`exp1_cell_sets.csv`（每个设置的电芯数和排除数）、`exp1_estimability.csv`、`exp1_leakage_audit.csv`。
- **实验 1，只在正式运行：** `exp1_predictions.csv`、`exp1_selected_hyperparameters.csv`、`exp1_pair_ci.csv`、`exp1_reversals.csv`、`exp1_h1_settings.csv`、`exp1_descriptive_rmse.csv`、`exp1_descriptive_excl_b1c18.csv`、`exp1_common_practice_theta080.csv`。
- **实验 3，开发和正式都有：** `exp3_labels.csv`、`exp3_samples.csv`、`exp3_features_{hust,mit,nca}.csv`、`exp3_windows.csv`、`exp3_method_estimability.csv`、`exp3_preprocessing_audit.csv`、`exp3_check2_traj_equivalence.csv`。
- **实验 3，只在正式运行：** `exp3_predictions.csv`、`exp3_pair_ci.csv`、`exp3_reversals.csv`、`exp3_comparisons.csv`、`exp3_check5_baseline.csv`、`exp3_descriptive_e1_by_t0.csv`、`exp3_common_practice_theta080.csv`。
- **实验 2，只在正式运行：** `exp2_agreement.csv`、`exp2_equivalence.csv`、`exp2_detection_power.csv`、`exp2_detection_summary.csv`。

## 6. 测试计划（阶段 2，只用合成数据）

1. 内层交叉验证：在合成数据上，`ElasticNetCV` + 40 个划分选出的 (α, λ)，与手写的逐 (α, λ) 平均 MSE 循环选出的结果相同。
2. 不泄漏：内层索引 ⊂ 外层训练折；标准化统计量只来自训练折；负对照能识别出人为制造的泄漏。
3. bootstrap：v4 的中位数版函数与 v2 的 `error_pair_ci` 逐位相同；平均值版在没有 NaN 时与直接计算相同；缺失比例规则正确。
4. 判定逻辑：手工构造的表格覆盖 H1、H2、H3 的每个分支和边界（恰好 5%、恰好 50%、1 个与 2 个 θ 点）。
5. 等效性分类的边界（区间端点恰好等于 ±δ）；检出概率在 1 倍和 2 倍标准误下等于 0.0284 和 0.266。
6. GB 的折循环与 RF 使用相同的训练行和预处理；可估计性规则相同。
7. 替代负对照：重写的函数在构造的"中位数恰好相等"情形下与 v3 的逻辑一致。
8. 开发模式：端到端跑一个合成的 MIT 型数据集和一个合成的 RUL 型数据集；开发模式的输出只包含白名单中的文件和列；在开发模式下调用评价函数会抛出异常。
9. 正式模式端到端（合成数据）：两次运行逐字节相同。

## 7. 方案中含糊、矛盾或无法照做之处（需要作者决定）

**Q1【矛盾，会导致检查 S 失败】检查 S 第 1 项的相对差 ≤ 1e-12。**

- `1_severson_features_mit124.csv` 是用 `float_format="%.12g"` 写出的，只保留 12 位有效数字。舍入造成的相对误差最大可达 5×10⁻¹²/首位数字。按存档文件中的数值估算，约 90% 的数值舍入误差的上界超过 1e-12。即使重算结果完全正确，这一项也几乎一定不通过。
- 另外，10 个电芯的 `IR_Min_2_100` 为 0，相对差没有定义。
- 可选做法：
  - (a) 把重算值用同样的 `%.12g` 格式化后，与 CSV 文本逐字比较；
  - (b) 改为相对差 ≤ 1e-11，值为 0 时用绝对差 ≤ 1e-12。
- 建议 (a)。两种做法都需要写修订文件。

**Q2【没写清】检查 S 第 2 项"四舍五入到 5 位有效数字"的写法。**

- 存档的准备脚本只计算了未舍入的最大差（5.0×10⁻⁴），准备报告中"舍入后差为 0"的那一步不在存档代码中。
- 建议：`float(format(x, ".5g"))` 与 Attia CSV 解析后的值完全相等。

**Q3【关键，没写清】H1 中"设置 = (规则, Q_ref, θ)"与"P 的电芯取决于定义对"之间的关系。**

- 同一个 (规则, Q_ref, θ) 可以属于两个定义对（例如 strict、nominal、0.90 同时属于 nominal–first 和 nominal–first5），两个定义对的共同电芯不同，因此 C1、C2 的检验结果可能不同。
- 可选做法：
  - (A) 以 (规则, 定义对, Q_ref, θ) 为单位，每个定义对内各检验一次，"至少一半"的分母相应变大；
  - (B) 每个设置只检验一次，用该定义自己的全部有标签电芯。但这与 §2.1"训练和评价都在这两种定义共同的电芯上进行"不一致，还需要另做一套拟合。
- 建议 (A)。

**Q4【没写清】δ 的取值依据。**

- 实验 1 的 first–first5："同一 θ、同一规则下 nominal 参考的基准值"用哪些电芯计算？
  - (a) nominal 在 first–first5 的共同电芯上的值（需要在这些电芯上额外拟合 nominal 的 variance 模型）；
  - (b) 取 nominal–first（或 nominal–first5）那一对中 nominal 的值；
  - (c) nominal 在其自身全部有标签电芯上的值。
- 实验 1 的 nominal–first 和 nominal–first5：δ 是否就用该对共同电芯上的 nominal 点估计？
- 实验 3：nominal 下 BASE 的 E1 中位误差取点估计；如果 nominal BASE 在某个 θ 不可估计，δ 无法定义，这次比较怎样计？建议记为"精度不足"并单独列出。

**Q5【没写清】H2 的分母。**

- Δ 的 90% 区间不可估计的比较（任一设置不可估计，或 bootstrap 缺失比例 > 5%），应当排除在分母之外，还是计为"精度不足"？
- 建议：排除，并单独报告数量。

**Q6【方案与准备报告不一致】实验 3 的方法集合。**

- 准备报告 §4.2 写"方法为 RF、GB、BASE"；方案 §4 写"方法……全部沿用 v2 和 v3，不做修改。只新增 GB"。
- 这会影响以下几项：
  - 检查 2 中的 TRAJ 等价是否适用；
  - 检查 4 中 WAFT 的审计（v3 修订 01 正是由 WAFT 的审计触发的）是否适用；
  - 检查 5 的"最佳非基准方法"在哪些方法中选。
- 可选做法：
  - (a) 运行 RF、TRAJ、WAFT、BASE、GB，直接调用 v2 的 `run_setting`，检查 0–4 的定义与 v2、v3 完全相同，判定只看 RF–GB；
  - (b) 只运行 RF、GB、BASE。
- 建议 (a)。还需要决定检查 5 的"最佳方法"是在 v2 的三种方法中选（与已知的 HUST 97.1%、MIT 71.3%、NCA 82.9% 可比），还是把 GB 也算进去。

**Q7【没写清】实验 3 检查 5 的 θ 范围。**

- 可选：(a) v2、v3 的整个网格（与已知结果的口径相同，但要在窗口外的 θ 也拟合）；(b) 只用窗口内的 θ。

**Q8【无法照做】实验 1 的检查 0 和检查 5。**

- 实验 1 没有 t₀（预测时刻固定为第 100 个循环），也没有 BASE。
- 建议：检查 0 和检查 5 只适用于实验 3；实验 1 用检查 1–4、6、S 和实验 1 的不泄漏检查。

**Q9【没写清】检查 S 未通过时，"实验 1 不参与判定"的后果。**

- 建议：H1 记为"无法检验（检查 S 未通过）"；H2 只用实验 3 计算，并在结果中注明。

**Q10【没写清】检出能力（§3.3）的范围和基准。**

- (a) 只算判定用的方法对，还是全部方法对？
- (b) E3 的指标是 C 指数，"基准误差的 10% 和 20%"不适用。E3 是只报告 1 倍和 2 倍标准误两档，还是整体不报告？H3 中"检出能力低"的中位数是否只用 E1 和 E2？
- (c) m 的基准取 nominal 下的基准值（与 δ 相同），还是每个设置各自的基准值？
- 建议：(a) 只算判定用的方法对；(b) E3 只报告标准误两档，H3 的中位数用 E1、E2；(c) 用 nominal，与 δ 一致。

**Q11【实现细节，方案没写】**

- 内层种子的"重复序号"从 0 还是从 1 开始（建议 0–9）；
- elastic net 的 `max_iter` 和 `tol` 用 sklearn 默认值（1000、1e-4），未收敛时只记录警告次数，不改设置；
- 内层各折不重新标准化（只用外层训练折的统计量，不涉及测试折）。

**Q12【没写清】"排除 `b1c18` 后重算"的范围。**

- 建议：只在 P 条件下，把 `b1c18` 从训练和评价中都去掉，重新拟合，重算 C1、C2 和 discharge–full 的区间；只作描述。C 条件也照此做一遍。

**Q13【没写清】实验 1 的 bootstrap 写法。**

- 建议沿用 v2：在 124 个电芯上抽样，不属于该集合的电芯记为 NaN，有效电芯 ≥ 5，缺失比例 ≤ 5%。

**Q14【没写清】H3 中所有数据集都因检查 0–5 被排除时的判定。**

- 建议：记为"无法检验"。

**Q15【没写清】H1"在同一定义对之间"的含义。**

- 建议：同一 (规则, 定义对)，即 strict 的 nominal–first 与 tolerance 的 nominal–first 不合并。

**Q16【约定之间的冲突】开发运行的范围。**

- 阶段 3 的指示是"只运行合理性检查 0–6"。但检查 5 是方法减 BASE 的置信区间，本身就是方法比较，与盲态冲突；检查 6 需要两次正式运行。
- 建议：
  - 开发运行只做检查 0–4、S、实验 1 的不泄漏检查和可估计性；
  - 检查 5 只在正式运行中计算；
  - 检查 6 在开发阶段只用两次开发运行的输出做一次确定性预演。
- 另外请确认：开发运行中为判断"能否拟合"而在内存中计算内层交叉验证 MSE，但不写出、不打印，这是否可以接受。如果不可以，实验 1 的可估计性在开发运行中只能按电芯数判断，"能拟合"留到正式运行再判断。

**Q17【没写清】实验 3 的 C 条件。**

- 建议：报告 E1、E2、E3 上 RF、GB、BASE 的点估计，以及按点估计得到的"表面排名变化"。

**Q18【没写清】实验 1 的 θ 网格。**

- 方案没有写明，准备脚本用的是 0.75–0.95。建议沿用。

**Q19【没写清】§3.1 一致比例包括哪些方法对。**

- 建议：包括全部方法对（discharge–full、RF–BASE、GB–BASE 等），分方法对报告，其中判定用的方法对单独列出。
