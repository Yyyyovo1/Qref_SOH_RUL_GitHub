# 后续描述性分析（follow-up analyses）

只读、只作描述的分析，没有新的判定规则。输入都是已提交的冻结输出；输出写到 `project/outputs/followup_analyses_v1/`。

| 脚本 | 内容 |
| --- | --- |
| `followup.py` | θ=0.80 下 MIT / HUST 的逐电芯核查；21 个 θ 下的有标签电芯数（`2_labeled_cells_by_theta.*`） |
| `followup2.py` | 停止 / 初始位置，可观测阈值窗口，等效阈值预测，v2 反转相对窗口的位置，带注释的图（`3_*`） |
| `plot_summary_annotated.py` | 规范验证四个陷阱的汇总图；复制自 `project/protocol_validation/`（`885d9e6`），只是 PDF 不写入创建日期 |
| `run_all.py` | 按顺序运行以上三个脚本 |

运行：`.venv/Scripts/python.exe project/followup_analyses/run_all.py`

- 这些脚本最初在 `reruns/protocol_validation_followup/`（未纳入版本控制）中编写和运行。迁入本目录时只改了输出路径，并让 PDF 不写入创建日期。
- 与 `reruns/` 中原结果的比对：CSV、JSON、PNG 逐字节相同；PDF 只有文档信息对象（创建日期）和随之变化的字节偏移不同，内容相同。
- 两次独立运行本目录脚本，全部 18 个输出文件逐字节相同。
