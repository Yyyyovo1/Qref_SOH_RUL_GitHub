# v4 实现说明：相对 PROTOCOL_v4.md 和修订 01 的澄清（未改变方案）

- **状态：** 作者已确认（2026-11-14）。本文件与 v4 代码一起，在开发运行和正式运行之前提交；开发运行的记录在正式运行前补在本文件末尾。
- **依据：** `PROTOCOL_v4.md`（`74f645a`）、`AMENDMENT_01.md`、`IMPLEMENTATION_PLAN_stage1.md` 及作者答复（2026-11-14）。
- 影响判定或合理性检查的内容都写在修订 01 中；本文件只记录实现细节。

## 1. 代码位置

- **代码包：** `project/rul_v4_core_claim/`，命名与 `rul_pilot_v2`、`rul_pilot_v3_tju` 一致。实现计划中写的 `core_claim_v4/` 不再使用，以免与存档目录 `project/v4_core_claim/` 混淆。
- **命令行：** `project/run_v4_core_claim.py`。
- **测试：** `project/tests/test_v4_*.py`，只用合成数据。
- **输出：** `reruns/v4/<运行名>/`。目录已存在时拒绝写入；先写入临时目录，完成后整体改名（沿用 v2）。`reruns/` 被 `.gitignore` 忽略，提交时用 `git add -f`，不改 `.gitignore`。
- **复用的代码：** v1、v2、v3、规范验证的代码只导入，不修改（清单见实现计划第 3 节）。实验 1 的特征通过 importlib 按路径导入存档的 `reruns/v4_prep/severson_features.py`，直接调用其中的 `cell_features` 和 `dq_features`。

## 2. 实验 1 的实现细节

1. **内层交叉验证（计划 Q11）：**
   - 重复序号 r = 0, 1, …, 9，第 r 次用 `KFold(n_splits=4, shuffle=True, random_state=20260928 + r)`，在外层训练折的电芯上分折，共 40 个划分。
   - 每个电芯只有一个样本，所以分折就是按电芯分。
   - `ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], cv=上述 40 个划分, n_jobs=1)`，λ 用默认路径（100 个值，eps = 1e-3），由外层训练折的数据计算。
   - `max_iter`、`tol` 用 sklearn 默认值（1000、1e-4）。未收敛时只记录警告次数，不改设置。
   - 内层各折不重新标准化：z 标准化的均值和标准差只用外层训练折计算，并用于外层训练折和外层测试折。公式与 `StandardScaler` 相同（ddof = 0，标准差为 0 时取 1），代码中直接写出（`exp1_models.standardize`），以便审计时逐位重算；测试核对两者结果相同。
   - 40 个划分的 MSE 等权平均，取最小值对应的 (α, λ)，在整个外层训练折上重新拟合。

2. **bootstrap（计划 Q13）：** 沿用 v2 的写法。
   - 抽样为 `draws_for(124)`，在全部 124 个 MIT 电芯上进行，电芯按 Cell_ID 字典序排列，与 v2 的 MIT 抽样相同。
   - 不属于该比较电芯集合的电芯记为 NaN；每次重抽样取 nanmean，有效电芯少于 5 个时这次重抽样记为缺失。
   - 缺失比例 > 5% 时，该区间不可估计。
   - 95% 区间取 2.5% 和 97.5% 线性分位数；修订 01 的 90% 区间取 5% 和 95% 线性分位数。

3. **排除 `b1c18` 的描述性结果（计划 Q12）：**
   - P 条件：把 `b1c18` 从训练和评价中都去掉，重新拟合三种模型，重算全部方法对（variance–discharge、variance–full、discharge–full）在两种定义下的区间和反转。
   - C 条件：同样去掉 `b1c18` 后重算点估计。
   - 这部分结果只作描述，写在 `exp1_descriptive_excl_b1c18*.csv` 中，不参与任何判定。

4. **θ 网格（计划 Q18）：** 0.75–0.95，步长 0.01，与准备脚本 `mit_label_windows.py` 相同。P 条件只取共同窗口内的点，C 条件只取 0.80。

5. **共同窗口：** 运行时重算，要求与存档的 `3_mit_common_windows.csv` 完全一致（另核对各设置的有标签电芯数与 `3_mit_labeled_share_eol_gt_100.csv` 相同）。正式运行中不一致就停止，不进入任何评价；开发运行只记录。

6. **检查 0（修订 01 §7）的做法：**
   - 用一个只暴露前 100 个循环的数据代理，重新计算 20 个特征：逐循环数据只到第 100 个循环（行 0–99），summary 数组截断到前 100 个元素。
   - 要求结果与完整数据下的计算逐位相同；同时记录逐循环数据被访问的最大行号，要求 ≤ 99。

7. **检查 S 的实现：**
   - 第 1 项按修订 01 §1：参照值从 CSV 读入时使用 `float_precision="round_trip"`。
   - 第 2 项按修订 01 §2。
   - 第 3 项：124 个电芯的 `Author_Cycle_Life` 与 Attia 三组寿命逐个相等，划分大小为 41/43/40。

8. **δ 的单独拟合（修订 01 §4，first–first5）：** 对每个 (端点规则, θ)，在 nominal 自身全部"有标签且 EOL > 100"的电芯上，用冻结的 MIT 5 折拟合 variance 模型，计算平均 APE。这次拟合的预测只用于 δ。各定义对的 δ 基准值及其来源（`pair_common_cells` 或 `nominal_own_labeled_cells`）写在 `exp1_delta_baselines.csv` 中。

## 3. 实验 3 的实现细节

1. **方法：** 直接调用 v2 的 `run_setting`（RF、TRAJ、WAFT、BASE）。GB 用单独的函数实现，折循环、训练行（每折中精确电芯的样本）、预处理 `_fold_preprocess(scale=False)` 和可估计性规则都与 `run_setting` 中的 RF 一行一行对应。
   - GB 的超参数运行时从 `outputs/soh_ml_design_v1/ml_model_hyperparameter_plan.csv` 读取，并与 `config.py` 中的值核对，不一致就停止。
   - 评价时沿用 v2 的 `Arrays`，另把 GB 的预测加入 `arr.pred`。

2. **计算顺序（修订 01 §6）：**
   1. 先只计算 RF、TRAJ、WAFT 相对 BASE 的区间；
   2. 按数据集计算检查 5，并与冻结的 `check5_baseline.csv` 核对；
   3. 核对通过之后，才计算其余方法对（包括 RF–GB、GB–BASE）。
   - 核对不通过时，第 3 步不执行。

3. **需要 bootstrap 向量的计算：** v2 的 `error_pair_ci` 只返回区间，不返回抽样向量。修订 01 的 Δ 和检出能力需要向量本身，所以 v4 写一个新函数：内部调用 v2 的 `per_cell_errors`、`_nanmedian_rows`、`_interval`，返回与 v2 相同的区间，另外返回抽样向量。测试要求它与 v2 的 `error_pair_ci` 逐位相同。

4. **替代负对照：** v3 修订 01 的替代负对照写在 v3 的 `run()` 里面，无法导入。v4 按相同逻辑写成一个函数，用测试核对它在 v3 的两种情形下与原逻辑一致：中位数不同；中位数相同但均值不同。

5. **C 条件（计划 Q17）：**
   - θ = 0.80，冻结 5 折。
   - 报告 E1、E2、E3 上全部方法（RF、TRAJ、WAFT、GB、BASE）的点估计，RF 和 GB 列在最前。
   - 另报告按点估计得到的"表面排名变化"，即两个设置下点估计的优劣方向相反。只作对照，不计算区间。

6. **共同窗口：** 运行时重算，要求与存档的 `4_exp3_windows_estimability.csv` 的 `In_Common_Window` 列完全一致。正式运行中不一致就停止，不进入任何评价；开发运行只记录。

## 4. 实验 2 的实现细节

- **§3.1 一致比例的范围（计划 Q19）：**
  - 包括实验 1 和实验 3 中 P 条件下的全部方法对，即 discharge–full，以及实验 3 中所有涉及 BASE、TRAJ、WAFT、GB 的方法对；实验 3 包括 E1、E2、E3。
  - 结果按实验、数据集、方法对、评价集分别报告，判定用的方法对另外单独列出。
  - 只作描述。
- **SE：** 为该设置方法差值 bootstrap 向量的标准差（ddof = 1），只用非缺失的重抽样计算。

## 5. 盲态的实现与测试（修订 01 §11）

- **评价函数的保护：** bootstrap、APE 和误差汇总、区间、反转、等效性、检出能力、判定这些函数，进入时都检查 `blinding.ALLOW_EVALUATION`。开发模式下该开关为假，函数一旦被调用就抛出异常。
- **输出白名单：** 开发模式只写白名单中的文件。写文件时检查列名，出现以下字样就报错：`Pred`、`Error`、`APE`、`RMSE`、`MSE`、`CI`、`Diff`、`Better`、`Reversal`、`Verdict`、`Alpha`、`L1`、`Coef`、`Lambda`。
- **警告的处理：** 开发模式下，sklearn 的 `ConvergenceWarning` 被捕获并只计数，不打印原文，因为原文中含有对偶间隙等数值。
- **测试：** 用合成数据跑一次开发模式，然后：
  1. 在测试里单独计算这次拟合的内层交叉验证 MSE、选中的 (α, λ)、系数和预测值，在全部输出文件、标准输出、标准错误和日志中搜索这些数值的字符串形式，要求一个都找不到；
  2. 在开发模式下直接调用评价函数，要求抛出异常；
  3. 核对输出文件名和列名都在白名单内。

## 6. 阶段 2 补充：编写代码时确定的实现细节（未改变方案和修订 01）

1. **模块：**
   - `config.py`：常数；
   - `prereg.py`：17 个存档文件的 SHA-256 核对；
   - `blinding.py`：盲态保护；
   - `severson.py`：特征、检查 0、检查 S；
   - `exp1_labels.py`、`exp1_models.py`、`exp1.py`：实验 1；
   - `exp3.py`：实验 3；
   - `exp2.py`：实验 2；
   - `verdicts.py`：H1–H3；
   - `pipeline.py`：串联；
   - `v2v3_agreement.py`：方案 §7 第 6 步；
   - `parallel.py`：多进程执行（§6 第 11 条）。
2. **存档核对：** 每次运行开始时，核对 `PREREG_RECORD.md` 和 `AMENDMENT_01_RECORD.md` 中列出的 17 个文件的 SHA-256，并核对这两个记录文件与 `config.py` 中的值相同。任何一项不一致就停止。
3. **输入文件核对：** 沿用 v2、v3 的做法，运行前后核对输入文件的 SHA-256（v2、v3 的全部输入，加上 `trapD_labels.csv`、GB 方案表、v2/v3 的 `check5_baseline.csv`、MIT 折和清单、存档的特征参照文件）。MIT 原始 `.mat` 文件（约 8 GB）和 Attia 数组不做整文件哈希；它们的内容由检查 S 逐项核对。
4. **实验 1 的可估计性：** 外层训练折少于 4 个电芯时（内层 4 折无法划分），该模型记为不可估计。按准备报告，每折训练电芯至少 87 个，预计不会出现。系数或预测值出现非有限值时，同样记为不可估计。
5. **警告：** 拟合时捕获全部警告，只按类别计数（`exp1_fit_warnings.csv`），不打印原文。正式运行和开发运行都这样处理。
6. **开发运行的输出：**
   - 只允许写 `pipeline.DEV_FILES` 中的文件（另允许 `exp3_features_<数据集>.csv`）。
   - 每个文件的列名都经 `blinding.forbidden_columns` 检查。被禁止的列名部分包括 `Pred`、`Error`、`APE`、`RMSE`、`MSE`、`CI`、`Diff`、`Better`、`Reversal`、`Verdict`、`Alpha`、`L1`、`Ratio`、`Coef`、`Lambda`、`Observed`、`Median`、`Mean`、`SE`、`Delta`、`Supported`、`Power` 等。
   - 例外只有逐个列出的列：输入特征名（如 `DeltaQ_Mean`、`*__level`、`*__slope`、`Prediction_Time_Raw_Cycle`）、哈希和布尔审计列、检查 2 的 `Max_Rel_Diff`、检查 S 的特征差异列。
   - 预测行数一栏命名为 `Fitted_Rows`。
7. **开发运行的子集：** `--experiments`、`--datasets`、`--thetas` 只允许在开发运行中使用。θ 为子集时不做共同窗口与存档的比较（记为跳过）。正式运行拒绝任何子集。
8. **测试**（`tests/test_v4_*.py`，只用合成数据）：
   - 内层划分；
   - elastic net 的选择（与逐点 ElasticNet 重算的交叉验证 MSE 对照）；
   - 标准化与 `StandardScaler` 一致；
   - 泄漏审计与负对照（用训练行冒充全部行时，负对照必须失败）；
   - 平均值 bootstrap 与逐次手算一致；
   - 返回向量的区间函数与 v2 的 `error_pair_ci` 逐位相同；
   - 重写的替代负对照与 v3 的原逻辑一致（三种情形）；
   - GB 的训练行和可估计性与 RF 相同；
   - 等效性分类的边界和归类顺序；
   - 检出概率常数；
   - H1、H2、H3 各分支和边界；
   - 检查 S 的容差；
   - 截断视图（合成的 MATLAB v7.3 结构 HDF5 文件，访问第 101 个循环时报错）；
   - 检查 5 的复现与不复现，以及 GB–BASE 不参与"最佳方法"；
   - 开发运行的盲态：重算 MSE、λ、系数和预测值，在全部输出、标准输出、标准错误和日志中搜索不到；
   - 开发运行只写白名单文件；
   - 两次正式运行（合成数据）逐字节相同；
   - 检查 5 不复现时停止。
9. **dev_run_1 之后的两处修复（代码错误，未改变方案和修订 01）：**
   - **检查 S 的身份列。** `check_s_features` 除了修订 01 规定的 20 个特征，还核对 `Batch`、`Severson_Split`、`Split_Position`、`Author_Cycle_Life` 四个身份列。**这一核对不属于修订 01 规定的检查 S 三项内容**，是代码额外加的一致性核对；但它不通过时，检查 S 也记为不通过。原代码把它们转成字符串比较；MIT 清单中的 `Author_Cycle_Life` 是浮点数，`"1852.0"` 与 `"1852"` 不等，124 个电芯全部被误判为不一致。现在数值列按数值比较，`Severson_Split` 仍按文本比较。
   - **开发运行的列名检查。** 实验 3 标签表中的 `Last_Observed_Index`（v1 检测器给出的最后一个保留观测的索引，属于标签信息）含有被禁止的词 `Observed`，导致 dev_run_1 在写文件前停止。现在只把 `Last_Observed_Index` 这一个具体列名加入允许列表，`Observed` 这个词本身仍然被禁止（测试核对 `Observed_A` 等仍被拦下）。新增测试：在合成数据集上实际运行 v1、v2、v3 的标签、样本、特征、拟合和审计代码（HUST、MIT、NCA 三种布局），把开发运行会写出的每张表都过一遍列名检查。
   - **命令行脚本：** 捕获盲态异常，打印 `RUL V4 STOPPED BY THE BLINDING GUARD` 和原因，以退出码 3 结束（原来是未捕获的异常）。
10. **收敛警告分开计数：**
    - sklearn 的 `ElasticNetCV.fit` 在同一次调用中完成内层交叉验证和最终整折重拟合，警告无法直接区分来源。
    - 做法：拟合完成后，用与 sklearn 1.7.2 `LinearModelCV.fit` 完全相同的参数（所选 α、λ，`precompute=False`，`copy_X=True`，`max_iter=1000`，`tol=1e-4`，`selection="cyclic"`）把最终重拟合单独重做一次，只用于计数，并核对它的系数与模型的系数逐位相同（`Final_Refit_Repeat_Identical`）。内层交叉验证的警告数 = 总数 − 重做的重拟合的警告数。预测值仍然来自原来的 `ElasticNetCV` 模型，计算不变。
    - `max_iter`、`tol` 等设置都没有改。
    - **输出：**
      - `exp1_fit_warnings.csv` 给出 `Inner_CV_ConvergenceWarning`、`Final_Refit_ConvergenceWarning`、elastic net 折拟合数、最终重拟合未收敛的折拟合数、重做结果不一致的折拟合数；
      - 审计表逐折给出 `Final_Refit_Converged` 和 `Inner_CV_Convergence_Warnings`；
      - 正式运行的 `decision.json` 中同样报告这些计数（`exp1_fit_warnings`）；
      - 排除 `b1c18` 的描述性重拟合另写 `exp1_descriptive_excl_b1c18_fit_warnings.csv`。
11. **并行化（只改变执行方式，不改变计算）：**
    - **实验 1：** `--workers N`。每个 (规则, Q_ref, θ, 模型, 电芯集合) 的外层 5 折拟合作为一个任务，任务按原顺序生成、去重后分发给 N 个进程（`spawn`），结果按提交顺序合并。内层交叉验证的设定随任务一起传入。无论 N 是多少，实验 1 的 BLAS 线程数都用 `threadpoolctl.threadpool_limits(1)` 固定为 1：特征计算、拟合和正式运行中的评价，在主进程和工作进程中都是如此。
    - **实验 3：** workers > 1 时只按数据集并行，最多 3 个进程，每个进程自己读取数据，并按 v2、v3 的方式拟合和评价，线程设置不变（默认值，与 v2、v3 相同）。正式运行的评价也按数据集分进程；工作进程中的评价由已受盲态保护的 `evaluate_datasets` 调用。
    - **等价性核对：**
      - 开发运行另写 `exp1_fit_hashes.csv`、`exp3_fit_hashes.csv`：每次拟合的预测向量和所选超参数的 SHA-256。
      - 这些表只含哈希，不含任何预测值、超参数或误差，从哈希无法还原数值，因此不违反修订 01 §11。列名 `Predictions_SHA256`、`Hyperparameters_SHA256` 作为具体列名加入允许列表。
      - 合成数据测试要求 workers = 1 与 workers = 8 在开发模式和正式模式下的全部输出逐字节相同（包括两张哈希表）。实验 3 的正式评价在测试中用两个合成数据集，实际走多进程路径。
      - 实验 3 按数据集并行的拟合路径需要真实数据加载器，只能在真实数据的开发运行中检验：两次开发运行都用并行代码，要求输出（包括 `exp3_fit_hashes.csv`）逐字节相同。
    - `run_info.json` 另记 `workers`、开始时间和耗时；检查 6 比较输出时不比较 `run_info.json` 和 `processing_log.txt`（沿用 v1 的 `compare_runs`）。
12. **修订 02（`AMENDMENT_02.md`，提交 `2fd53bb`，Zenodo 10.5281/zenodo.24248021）的实现：**
    - 只改 `exp1_labels.check2` 中容差规则部分的判据。严格规则部分（v2 的 `_physical_equivalence`，标签完全一致）和实验 3 的检查 2 不变。
    - 逐行判据写成 `tolerance_row_ok`：
      - 标签类型必须相同；
      - 有标签的行 |ΔEOL| / EOL ≤ 1e-9（`TOLERANCE_EOL_REL_TOL`）；
      - 无标签的行两边的 EOL 都为 NaN。
    - 按修订 02 §7，检查结果中报告：比较行数、阈值不同的行数、按原标准（完全一致）的不一致行数、标签类型变化的行数、相对差超过 1e-9 的行数、无标签行中不同为 NaN 的行数、最大 |ΔEOL| 和最大相对差。
    - `check2` 增加只供测试使用的参数 `alt_threshold`。
    - **新增测试（合成数据）：**
      - 逐行判据的各种情形；
      - 阈值统一偏移 1 ulp 时判为通过，此时原标准判为不通过；
      - 阈值相对偏移 1e-6 时，相对差超过 1e-9，判为不通过；
      - 阈值放大 1.2 倍时，标签类型改变，判为不通过。
    - 存档文件核对（`prereg.py`）仍只核对修订 01 时的 17 个文件，这次没有把修订 02 加进去，以免 `prereg_hash_check.csv` 与 dev_run_2 不同。是否加入由作者决定。
13. **正式运行前的准备（本条对应的提交说明为"正式运行前的准备：存档哈希核对加入修订 02，运行记录加入代码提交号；未运行正式分析"）：**
    - **存档哈希核对加入修订 02：**
      - 运行开始时核对的存档文件由 17 个增加到 18 个，新增 `project/v4_core_claim/AMENDMENT_02.md`（SHA-256 `e1f1937f…2e7bc2`）；
      - `AMENDMENT_02_RECORD.md` 也列入记录文件，并核对与 `config.py` 一致；
      - 任何一个文件不匹配都停止运行，与原来一样。这一项替代 §6 第 12 条最后一句"仍只核对 17 个文件"的做法。
    - **`run_info.json` 新增三项：**
      - `code_commit`：运行开始时的 `git rev-parse HEAD`；
      - `worktree_clean`：`git status --porcelain` 的输出是否为空（`.gitignore` 忽略的文件不算，因此 `reruns/` 下的输出不影响）；
      - `amendment_02_commit`：`2fd53bb`。
    - **工作区检查：** 正式运行中，如果工作区不干净，在存档核对之后、读取任何数据之前停止，命令行给出 `WORKTREE_NOT_CLEAN` 和未提交的文件列表。开发运行只记录，不停止。git 不可用时记为不干净。
    - 日志的第一行加上修订 02 的提交号（日志不参与检查 6 的比较）。
    - **新增测试：**
      - 本仓库的存档哈希表为 18 行且全部匹配；
      - 修订 02 的哈希不匹配时，运行在读取任何数据之前停止；
      - 正式运行在工作区不干净时停止，且没有调用任何拟合；
      - 开发运行在工作区不干净时照常进行，并在 `run_info.json` 中记录。

## 7. 开发运行记录

### dev_run_1（代码 `c16fab7`，2026-11-14 18:56–19:35，失败）

- **命令：** `python run_v4_core_claim.py --dev --output-dir ../reruns/v4/dev_run_1`，串行。
- **经过：**
  - 存档核对通过；实验 1 的标签、检查 1–3、特征、检查 0、检查 S 和全部拟合完成（拟合约 31 分钟）；
  - 实验 3 的三个数据集拟合完成（约 7 分钟）；
  - 写出实验 3 标签表时，列名检查拦下 `Last_Observed_Index`，程序停止，临时目录没有改名为输出目录。
- **已写出的文件：** `prereg_hash_check.csv`、`dataset_checks.csv` 和实验 1 的 13 个文件。没有 `sanity_checks.csv`，没有任何实验 3 的文件。临时目录改名为 `reruns/v4/dev_run_1_failed_c16fab7/` 保留，不参与任何判定。
- **查看过的内容：**
  - 检查表和审计表：`dataset_checks.csv`、`exp1_checkS_*.csv`、`exp1_check0_cycles.csv`、`exp1_leakage_audit.csv`（布尔列）、`exp1_estimability.csv`、`exp1_fit_warnings.csv`；
  - 标签、比例和窗口：`exp1_common_windows.csv`、`exp1_label_shares.csv` 与存档准备文件的对比，`exp1_cell_sets.csv` 的电芯数。
  - 没有查看、也没有计算任何误差、区间、比较或判定。
- **发现的两处代码错误：**
  1. 列名检查误拦 `Last_Observed_Index`；
  2. 检查 S 的身份列按字符串比较，`Author_Cycle_Life` 被误判不一致，`dataset_checks.csv` 中实验 1 因此显示为 `CHECK_S_FAILED`。
  - 按数值比较，124 个电芯全部相同；检查 S 的三项本身全部满足（特征最大相对差 4.6×10⁻¹²，`Qdlin` 124/124，寿命 124/124）。
  - 修复见 §6 第 9 条。
- **其他观察：** 实验 1 共记录 320,191 次 `ConvergenceWarning`，当时没有区分内层交叉验证和最终重拟合，因此新增 §6 第 10 条的分开计数。

### dev_run_2、dev_run_3（代码 `e379d25`，2026-11-14）

- **命令：** `python run_v4_core_claim.py --dev --workers 16 --output-dir ../reruns/v4/dev_run_2`，dev_run_3 相同。两次先后进行，各自完整运行。
- **耗时：** dev_run_2 549.1 秒，dev_run_3 547.8 秒。其中实验 1 的拟合约 4 分钟（285 个任务，16 个进程；dev_run_1 串行约 31 分钟），实验 3 的拟合约 5 分钟（3 个进程）。
- **两次的比较：** 除 `run_info.json` 和 `processing_log.txt` 外的 28 个文件逐字节相同，包括 `exp1_fit_hashes.csv` 和 `exp3_fit_hashes.csv`。这是检查 6 的预演。
- **检查结果：**
  - **实验 1：**
    - 通过：检查 0（124/124，访问的最大行号 99）、检查 1（严格规则 7812 行一致，43/43 在 ±1 以内；容差规则 7812 行与 `trapD_labels.csv` 一致）、检查 3（0 处违反）、检查 4（不泄漏各项）、检查 S（三项）、共同窗口与存档一致；可估计性 456/456，每个设置 112–124 个电芯。
    - **检查 2 未通过：** 严格规则 5208/5208 一致；容差规则中 427 行阈值不同（最大 1.1×10⁻¹⁶ Ah），其中 263 行 EOL 改变，全部为 `INTERPOLATED`，标签类型零变化，最大 |ΔEOL| 1.19×10⁻¹² 个循环，最大相对差 3.7×10⁻¹⁵。按当时的代码，实验 1 记为未通过检查 0–4。处理见修订 02 和 §6 第 12 条。
  - **实验 3：**
    - HUST、MIT、TJU-NCA 的检查 0、1、2、3、4（修订后和原规则）全部通过；GB 超参数与方案表一致；共同窗口 234 行与存档一致。
    - 检查 4 中替代均值对照的使用次数为 0。
    - TRAJ 等价的最大相对差：HUST 1.8×10⁻¹³、MIT 1.4×10⁻¹⁴、NCA 4.4×10⁻¹⁶。
  - **实验 3 可估计性：** RF、TRAJ、WAFT、BASE 共 936 行，与冻结的 v2、v3 `method_estimability.csv` 逐行相同（是否可估计、原因、样本数、TRAJ 回退比例和缺失比例）；GB 的可估计性和原因与 RF 完全相同。
- **收敛警告（实验 1）：**
  - 内层交叉验证 320,187 次；
  - 最终整折重拟合 4 次，即 950 次 elastic net 折拟合中有 4 次重拟合未收敛；
  - 为计数而重做的重拟合与原模型系数 950/950 逐位相同；
  - 两类合计 320,191 次，与 dev_run_1 的总数相同。
- **实验 3 预测哈希与冻结结果的比对：**
  - dev_run_2 的 `exp3_fit_hashes.csv` 中，RF、TRAJ、WAFT、BASE 在每个 (数据集, θ, Q_ref) 下的预测哈希，与冻结的 `project/outputs/rul_ranking_pilot_v2/predictions.csv`（HUST、MIT）和 `project/outputs/rul_ranking_v3_tju/predictions.csv`（NCA）按同样方式计算的哈希（按样本顺序，float64，`float_precision="round_trip"` 读入）比对：936/936 相同（HUST 252、MIT 252、NCA 432）。
  - 2026-11-15 在写入本说明前重算了一遍，结果相同。
  - GB 没有参与比对。只比较哈希，不读出预测值，不计算任何误差或比较。
  - 这说明 v4 实验 3 中 v2、v3 原有方法的预测与 v2、v3 的冻结预测逐位相同。
- **查看过的内容：** 检查表、可估计性表、审计表、警告计数、共同窗口、哈希表和两次运行的逐字节比较；为诊断检查 2，在标签层面重算了容差规则（只涉及阈值和标签）。没有查看或计算任何误差、区间、预测值、超参数数值、比较或判定。
- **代码在这两次运行之后的改动：** 只有修订 02 对应的检查 2 判据（§6 第 12 条）。

### dev_run_4、dev_run_5_exp1_w1（代码 `7996e88`，2026-11-15）

- **dev_run_4：**
  - **命令：** `python run_v4_core_claim.py --dev --workers 16 --output-dir ../reruns/v4/dev_run_4`；完整运行，16 个进程，耗时 496.5 秒。
  - **结果：** 全部检查通过，包括按修订 02 判定的实验 1 检查 2。
  - **实验 1 检查 2：** 严格规则 5208/5208 一致；容差规则 427 行阈值不同，按原标准 263 行不一致，标签类型变化 0，相对差超过 1e-9 的行 0，无标签行不同为 NaN 的行 0，最大 |ΔEOL| 1.19×10⁻¹² 个循环，最大相对差 3.7×10⁻¹⁵。
  - **与 dev_run_2 的比较：** 除 `sanity_checks.csv`、`dataset_checks.csv`、`run_info.json` 和日志外，其余 26 个文件逐字节相同，包括两张哈希表。两个不同的文件中只差两行：`sanity_checks.csv` 中实验 1 检查 2 一行（`Pass`、`Detail`），以及 `dataset_checks.csv` 中实验 1 一行（`Passes_Checks` 由 False 变为 True，`Reason` 变为空）。
- **dev_run_5_exp1_w1：**
  - **命令：** `python run_v4_core_claim.py --dev --experiments 1 --workers 1 --output-dir ../reruns/v4/dev_run_5_exp1_w1`；只跑实验 1，单进程，耗时 1931.0 秒，与 dev_run_4 同时进行。
  - **结果：** 实验 1 的全部检查通过。
  - **与 dev_run_2 的比较：** `exp1_fit_hashes.csv` 与 dev_run_2 逐字节相同（SHA-256 `99d4712c…f510ee`），说明在真实数据上，实验 1 的 285 个拟合任务的预测和超参数不随进程数变化。其余 14 个实验 1 文件和 `prereg_hash_check.csv` 也与 dev_run_4 逐字节相同。
- **查看过的内容：** 检查表、可估计性、哈希和逐字节比较。没有查看或计算任何误差、区间、预测值、超参数数值或比较。

### dev_run_6（代码 `01061f3`，2026-11-15）

- **命令：** `python run_v4_core_claim.py --dev --workers 16 --output-dir ../reruns/v4/dev_run_6`；完整运行，16 个进程，耗时 469.6 秒。
- **运行记录：** `run_info.json` 中 `code_commit` 为 `01061f3dafa5564559ec8d352c26dec68e17c02b`，`worktree_clean` 为 true，`amendment_02_commit` 为 `2fd53bb`。
- **存档核对：** `prereg_hash_check.csv` 为 18 行，全部 `Match = True`；原有 17 行与 dev_run_4 相同，新增一行为修订 02。
- **与 dev_run_4 的比较：** 参与比较的 28 个文件（不含 `run_info.json` 和日志）中，只有 `prereg_hash_check.csv` 不同。其余 27 个文件逐字节相同，包括两张哈希表、`sanity_checks.csv` 和 `dataset_checks.csv`。全部检查通过。
- **查看过的内容：** 检查表、存档核对表、`run_info.json` 和逐字节比较。没有查看或计算任何误差、区间、预测值或比较。

### 正式运行前补充的测试（只加测试，未改计算代码）

- **排除 `b1c18` 的过滤**（`DESCRIPTIVE_EXCLUDED_CELL` 设为合成数据中真实存在的电芯）：
  - 排除后的拟合和评价中都不含该电芯；
  - 原来包含该电芯的设置，电芯数正好少 1，其余设置不变；
  - C 条件中只有该电芯所在子集少 1 个有标签电芯；
  - 主分析的输出（区间、反转、H1 计数单位、预测、实验 2、实验 3、`decision.json`）与排除一个不存在的电芯时逐字节相同，只有 `exp1_descriptive_excl_b1c18*` 文件不同。
- **共同窗口与存档不一致**（分别模拟实验 1 和实验 3）：
  - 正式运行在任何评价之前停止，没有调用任何评价函数；
  - `decision.json` 只写停止原因；
  - 除开发运行允许的文件外，只多出 `decision.json`、`run_info.json` 和日志，没有写出任何比较文件。
- 这些测试第一次运行时有失败，原因都在测试本身的预期（C 条件表每个模型一行；`c01` 所在的设置数算反；把文件名中的 "equivalence" 当成比较文件）。改正测试后全部通过，代码没有改动。
