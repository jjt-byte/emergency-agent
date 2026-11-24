# 测量与证据说明

本目录归档自原交付材料。GitHub 整理副本的新增启动修复和本机测试结果见 [仓库整理与验证说明](../docs/仓库整理与验证说明.md)。

历史记录在 `historical/`，本次重新测量在 `release-cpu-benchmark-20261124.json`。不把历史数据复制到运行时 `data/` 充当新的验证结果。

| 文件 | 性质 | 口径 |
|---|---|---|
| historical/llama-benchmark.json | 2026-10-30 原机测量 | CPU，8 线程，pp256/tg128，各 2 次；token/s，不含 Agent 工具 |
| historical/person_detection_validation.json | 历史样例 | 六张图，不含完整 GT 标注框，不计算准确率 |
| historical/fire_smoke_validation.json | 历史样例 | 3 正 1 负；正例中仅 1 张检出，存在明确漏检 |
| historical/person_npu_comparison.json | 2026-11-24 原机硬件记录 | 每图预热 1 次、测量 3 次、取中位数，再平均六图；状态 needs_review |
| historical/photo-suite-smoke-test.json | 流程验证 | 照片拼接，36 帧；不能作为真实视频精度 |
| historical/real-video-test-summary.json | 人员实拍 / 烟火负例 | 6 帧；cycle_seconds 为最后一帧状态，不是全片均值 |
| historical/agent-acceptance-summary.json | 四个历史任务摘要 | 各自端到端时延，输入和轮次不同 |
| release-cpu-benchmark-20261124.json | 本次补测 | 每模型一张图；首轮一次 + 5 次热调用；外层 perf_counter |
| release-verification.json | 本次交付验收 | 软件测试、真实模型部署、单次 Agent 链路与未实测范围 |

本次热调用均值：人员 0.243522 秒，烟火 0.248329 秒；加载分别 3.156931 / 3.190656 秒。没有温度、功耗、长期负载和完整测试集，因此不提供能效、mAP、事件级漏报率、p95/p99 或 SLA。

`provenance.json` 保存历史原文件 SHA256 与导出变换说明。历史绝对路径替换为 `<ORIGINAL_PROJECT>` 等标识；数值与检测框未改写。Agent 摘要移除了用户对话、图片和生成正文，仅保留必要时延、状态和工具名。历史文件中的 passed/fingerprint 仅记录当时状态，不是可导入的新机器 ready marker。

`models_manifest.json` / `samples_manifest.json` 对应独立交付包；包含完整相对路径、字节数与 SHA256。`original_source_sha256.json` 是整理开始前保存的 57 个原版代码文件指纹，可检查原目录仍与整理前一致。

复测：使用 config.json 中指定的推理解释器运行 `scripts/benchmark.py --repeats 5`。输出默认写 `data/benchmark-current.json`，不会覆盖已归档数字。完整 Agent 需单独计时；两个检测器并发的视频周期也需另测，不能将独立热调用均值相加或取最大值就当作实际结果。
