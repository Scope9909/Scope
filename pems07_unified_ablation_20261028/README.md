# PEMS07 同协议分阶段资源与精度核算（seed 1）

本表使用相同的 PEMS07 12→12 划分、batch 32、验证集拟合校准表、测试集评价和冻结 Student checkpoint。状态文件均按包含索引与解码信息的完整文件计字节。Teacher 行仅是**基础预测参考**，不是最终 Hist2Res Teacher 的完整检索链路；不能把 Teacher→Student 的全部差异归因于低秩投影。

| 配置 | 参数量 | Test raw MAE | Test 校准后 MAE | checkpoint (B) | 检索状态 (B) | 校准表 (B) | 三资产合计 (B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 稠密 Teacher，基础预测 | 1,059,821 | 19.0012 | 18.7532 | 4,266,648 | 0 | 2,293,093 | 6,559,741 |
| 低秩 Student，基础预测 | 703,330 | 19.0210 | 18.7793 | 2,836,469 | 0 | 2,292,500 | 5,128,969 |
| 同一 Student，全节点精确检索 | 703,330 | 19.1490 | 18.7459 | 2,836,469 | 10,326,391 | 2,156,991 | 15,319,851 |
| 同一 Student，SRM 精确检索 | 703,330 | 18.7355 | 18.5091 | 2,836,469 | 83,063 | 2,216,230 | 5,135,762 |
| 最终 DARC，SRM + 冻结 Q8 编码 | 703,330 | 18.7381 | 18.5115 | 2,836,469 | 80,752 | 2,218,527 | 5,135,748 |

同一 Student 内，全节点精确检索→SRM 精确检索使校准后 MAE 降低 0.2368，三资产合计减少 10,184,089 B；这反映了当前 PEMS07 节点选择的收益，不证明 SRM 在其他数据集也改善精度。SRM 精确检索→最终 Q8 编码使检索文件减少 2,311 B，校准表增大 2,297 B，所以三资产净节省仅 **14 B**，而校准后 MAE 增加 0.0024。与全节点精确检索相比，最终配置的三资产合计减少 10,184,103 B（66.48%），但这一比例仅适用于本 PEMS07 实例和下述三类资产，不能称为完整部署系统的总体压缩率。

## 口径与限制

- 三资产合计仅为 checkpoint + Test 检索状态文件 + 该配置由 Validation 拟合的校准表；不含运行库、代码、缓存和加载峰值内存。不同配置的校准表单独拟合，参数形式和超参数相同。
- 全节点精确状态由既有全节点状态逐值无损转为 `uint16`，并使用已在稀疏精确状态上固定的相同逐字段 `.alsc` 规则编码。它是一个受控同容器对照，不宣称是对全节点状态重新优化后的最小文件。
- 全节点与 SRM 精确状态的测试 MAE 分别逐值复现先前原始状态评估；最终 Q8 行引用此前已冻结的测试结果，未根据本表重新选点、重新搜索 Q8 或调校测试指标。
- Teacher 基础预测的 checkpoint 仍包含其原有检索模块参数，虽然本行推理未调用它们。因此 Teacher 行不应作为“纯低秩预测头”的参数消融；同一 Student 的后三行才是严格的 SRM/编码阶段对照。

## 可核验输出

- [全节点精确状态评估](pems07_full_exact_metrics.json)；[全节点状态构建与评估脚本](run_full_exact.py)
- [Teacher 基础预测评估](pems07_teacher_base_metrics.json)；[Teacher 评估脚本](run_teacher_base.py)
- 四个阶段的 `pems07_*_calibrator.json`、`pems07_*_calibrator.npz`；[校准表导出脚本](export_stage_calibrators.py)
- Student 基础预测、旧全节点和 SRM 的既有结果：`../srm_test_once_20261014/pems07_s1_{backbone,full,sparse}_test_once.json`
- 最终 Q8 结果：`../contrib3_index_codec_20261020/packages/online/adaptive_lossless_test_20261023_v1/pems07_metrics.json`；最终校准表：`../contrib3_index_codec_20261020/packages/online/frozen_calibrators_20261024_v3/pems07_calibrator.npz`
