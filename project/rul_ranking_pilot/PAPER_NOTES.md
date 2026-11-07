# 写论文时使用的备注（不影响实验）

## 1. HUST 与 MIT 的电芯型号和容量的放电倍率

核对于 2026-11-06。

**HUST（Ma et al., 2022, *Energy Environ. Sci.* 15, 4083–4094, doi:10.1039/D2EE01676A）**

原文 “Experimental data” 和 Methods 两节写明：

- 电芯为 77 个 LFP/石墨 **A123 APR18650M1A**，标称容量 1.1 Ah、标称电压 3.3 V，在 30 °C 下用 Neware CT-4008 测试仪循环。
- 充电：5C 恒流从 0% 充到 80% SOC，再以 1C 恒流充到 3.6 V，然后恒压充到 100% SOC，截止电流 C/20。
- 放电：四个恒流阶段。前三段的倍率随协议变化，分别覆盖 100→60%、60→40%、40→20% SOC；最后一段从 20% 放到 0% SOC，论文写的是 1C，截止电压 2 V。
- 循环持续到最大容量首次降到标称容量的 80%。共 80 个电芯，其中 3 个因突发故障被排除，剩 77 个。
- 来源说明：RSC 页面对自动访问返回 Cloudflare 验证页，我没有绕过；上述内容取自 Semantic Scholar 上该文的 PDF 全文。Mendeley Data 页面（doi:10.17632/nsc7hnsg4s）没有写电芯型号，许可为 CC BY 4.0。

**本研究中 HUST 容量的含义**

- 容量 = 整个多段放电的总放电容量，即从第 1 段开始到最后一段结束放出的容量。在 4 个电芯上逐循环核对过，与原始数据中的 `dq` 字段一致（`outputs/hust_4cell_protocol_validate_v1/hust_4cell_capacity_semantics_validation.csv`）。
- 已记录的差异：原始数据中最后一段的电流约为 2C（≈ −2.2 A），而论文写的是 1C（见 `hust_paper_vs_raw_protocol_audit.csv`）。

**MIT（Severson et al., 2019）**

- 电芯也是 **A123 APR18650M1A**，1.1 Ah。
- 每个循环的容量为汇总数据中的放电容量。按 Severson Methods 的写法，放电是 4C 恒流-恒压放电到 2.0 V，截止电流 C/50。数据中逐循环放电电流的中位数为 −4.0 C。
- Ma et al. (2022) 引用 MIT 数据时写的是“uniform CC discharge of 4C to 2 V”，没有提到恒压段。

**可以写进论文的观察（需要谨慎表述）**

两个数据集用的是同一型号电芯，但实测参考相对标称容量的方向相反：

- HUST：first 和 first5 都比标称高约 +7.4%（77/77 个电芯）；
- MIT：都比标称低，中位数分别为 −0.031 Ah 和 −0.028 Ah（124/124 个电芯）。

两者的容量测量方式不同：MIT 是单一 4C 放电（带恒压段），HUST 是多段放电（最后一段的倍率较低）。放电倍率的差异是一个**可能的**解释，但本项目没有直接检验这一点。另外，两者的测试仪、批次和充电协议也都不同。

## 2. MIT 合并点的容量变化与批次页说法不符

见 `project/mit_repro/DATA_INVENTORY_NOTES.md` 第 2 条。
