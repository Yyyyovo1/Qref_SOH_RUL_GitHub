# v4 预注册存档记录

| 项目 | 内容 |
| --- | --- |
| DOI | [10.5281/zenodo.24222904](https://doi.org/10.5281/zenodo.24222904) |
| 记录页 | https://zenodo.org/records/24222904 |
| 发布日期 | 2026-11-14 |
| 许可 | CC BY 4.0，公开 |

存档后未修改上述文件；DOI 不写回方案本身。

## 存档文件（15 个）

路径为相对于仓库根目录的路径。SHA-256 为存档前在本地计算；MD5 与 Zenodo 记录页给出的 MD5 逐个一致。

| 路径 | 字节数 | SHA-256 | MD5 |
| --- | --- | --- | --- |
| `project/v4_core_claim/PROTOCOL_v4.md` | 17856 | `f14204f90df78a5cfacc024699b3d2b3399b536de99f1a6751fa5a7ec65ad9b9` | `45715d114fcda6842211bec6ea2e0024` |
| `reruns/v4_prep/1_feature_comparison_summary.json` | 2567 | `d4c9f22a53ba073f7509d9023705aab85762151d91bb58da07acecde48a23746` | `9e363534baf613283e396775b71fd80d` |
| `reruns/v4_prep/1_feature_comparison_vs_attia2021.csv` | 14680 | `ccd19bd69d77a52639544b48395292b43872d82275f7e86a988881ffa97149d1` | `cbdaab05e651431c4792b3b8ff5af0bd` |
| `reruns/v4_prep/1_severson_features_mit124.csv` | 36182 | `8c5cbb8c3afe969bddf49acd8fc191c699cd05ee3f6ad98ec443a4dfa97e1ec5` | `cd4657fe3bfdabdcf887111efa337e62` |
| `reruns/v4_prep/3_mit_common_windows.csv` | 588 | `93ffac0316a990a1a9dd32a3848cc75b370d720c89b802f81f8f461170a2475c` | `5f35da2592b59da43bde0d11fda8605a` |
| `reruns/v4_prep/3_mit_estimability_by_pair_theta.csv` | 7740 | `c2141253d55ecde62a3abbfafdde52d670e11b703b7f2f7698717691be1c20c2` | `dc0abd48799947177a306b5e5dd62742` |
| `reruns/v4_prep/3_mit_label_summary.json` | 143 | `6bb5750305514893501a627eabbbca656a68834855676c32ef86e9d79bb48d04` | `e071a595fd655b0361bf33ffa9b6f843` |
| `reruns/v4_prep/3_mit_labeled_share_eol_gt_100.csv` | 4957 | `df1a6ed2516a29a4c6419fedb81ac9620204d10b337b03832c6d88fec898d178` | `a1b03229625b01b05f8b91eb2996c926` |
| `reruns/v4_prep/3_mit_theta080_original_split_coverage.csv` | 267 | `f568d1b75ac79111cccfadf208313151704a5d893b32c2d26c070fd9ec9a6f1c` | `8c2c818012243055a0d9ffaf669c9edf` |
| `reruns/v4_prep/4_exp3_window_summary.csv` | 455 | `aed142dbb980c3f1fb5977c670043b31f49b366e14eb8b19eb491943387ba2c0` | `03f93bfeb1808ca4a4243a9a730558d3` |
| `reruns/v4_prep/4_exp3_windows_estimability.csv` | 10802 | `b29b82dafb1d2082d9a1e121ed2923600f31b5874a18dc3edf5374c06cfde7e9` | `47c1ca27723984af22529b2fdc3aca94` |
| `reruns/v4_prep/V4_PREP_REPORT.md` | 11495 | `6ed7926ef48c4d317f92b6bbcabb8ba371ca8c1e97beeaa1e0d936d13a593eaf` | `01294893934bc7b723a7c8dd4059c820` |
| `reruns/v4_prep/exp3_windows_estimability.py` | 4317 | `163505b2b4b64ce0c3433cca01c4d9033f169049cf023548c71aa889f827e7fc` | `901a9845a1e2c4e03d253deb26e9c56b` |
| `reruns/v4_prep/mit_label_windows.py` | 5357 | `c75b9022934abd05e6fb841fddf0cb1d4c027da999a7c8526d82cc6f345fbb0c` | `83a9b8c4198235d00c08d6cf05c33e6b` |
| `reruns/v4_prep/severson_features.py` | 8318 | `81a817723239e13db53562084dc712f35f72dcc4b4f49a6d86cb742872fab209` | `7555e06352ac0b9e7271b7e5517998de` |

## 核对方法

```bash
sha256sum <路径>
md5sum <路径>
git show HEAD:<路径> | sha256sum
```
