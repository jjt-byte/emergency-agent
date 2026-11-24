> 历史开发文档，保留原版记录；其中旧路径、版本状态和指标不作为当前交付结论。部署以仓库 README 为准，实测结论以《技术白皮书》及 benchmarks 为准。

# 应急先行：CIX NPU 模型迁移

更新日期：2026-11-24。`.cix` 是 CIX NPU 的模型编译格式；person、smoke、fire 是检测类别，两者分别由编译工具和训练权重决定。

最新实测：板卡普通终端已成功运行 YOLOv8s NPU，但六人图只检出四人，验收返回 `needs_review`，当前人员检测仍为 DAMO CPU。另发现“四人图”的参考只统计前景，背景还有人员，需要先复核参考集。下一步先执行下文诊断命令；详细证据和分支处理见 [NPU人员检测排查与下一步](NPU人员检测排查与下一步.md)。

## 已找到的本地资源

| 用途 | 当前资源 | 可做的事情 |
| --- | --- | --- |
| 人员检测 | `models/person/cix-yolov8s/yolov8s.cix`；DAMO 编译结果 `data/cix-compiled-20261124/person/damo_person.cix` | YOLOv8s 有独立适配器但此前验收未通过；DAMO `.cix` 已编译，项目尚无匹配的 NOE 适配器 |
| 其他 YOLO | `/home/orangepi/.cache/modelscope/models/cix--ai_model_hub_yolo_series/snapshots/master/models/ComputeVision/Object_Detection/` | 下载了 YOLOv8n/s/m/l、YOLO11 等，输入、输出和后处理须逐模型核对 |
| 烟火检测 | CPU 权重 `models/fire-smoke/damoyolo-smokefire/damoyolo_tinynasL25_S_smokefire.pt`；DAMO 编译结果 `data/cix-compiled-20261124/fire_smoke/damo_fire_smoke.cix` | `.cix` 已编译且文件校验通过；NOE 推理、量化精度和项目接入尚未验证 |
| 语言模型 | `/home/orangepi/models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf` | 继续用 llama.cpp；本地 CIX 的 Qwen2.5-3B 文档列出 CPU、Vulkan GPU、MNN 路线 |
| CIX 编译器 | `/home/orangepi/Downloads/cix_noe_sdk_26_q2_release/noe_sdk_26_q2_release/cixbuilder-6.1.3753.3-cp310-none-linux_x86_64.whl` | 此版本需要 x86_64 Linux 与 Python 3.10；板卡是 aarch64，不能直接安装此 wheel |

项目 YOLOv8s 的 SHA256 与 `source.json` 一致。输入为 `[1,3,640,640]`，输出为 `[1,84,8400]`，目标是 `X2_1204MP3`，适配器只取 COCO 类别 0（person）。该 COCO 检测模型没有 smoke/fire 类别，更改显示名称或配置不会增加烟火识别能力。YOLO-World 等其他模型也需核对其实际导出类别，不能仅凭模型名称视为可用的烟火模型。

## 人员检测：先诊断，再验收切换

在板卡普通终端执行：

```bash
cd /home/orangepi/emergency-agent
ls -l /dev/aipu
./diagnose-person-npu.sh
```

诊断脚本先比较旧版 FFmpeg 与新版进程内图片预处理，再在真实 NPU 上对比四种输入方式，保存原始张量、检测框、置信度与 NMS 阈值表。输出目录为 `data/npu-diagnostics/<时间戳>/`，脚本结束时打印 `report.html` 的绝对路径。诊断不会切换配置；输入方式和阈值对照用于定位原因，不能按这几张图片的人数挑参数后直接当作准确性结论。

`test-person-npu.sh --activate` 是后续的验收和切换入口。它会比较 DAMO CPU 与 CIX YOLOv8s NPU，预热后各测三次，记录人数、框和耗时。当前参考集中的 `000000013291` 已明确标记为不完整，即使人数碰巧一致也会阻止自动切换。须复核、补全或替换参考样例，并更新验收参考集后，再运行该入口。完整参考下六张 YOLO 样例人数全部正确且多次稳定才备份 `config.json` 并启用 `person_detection.backend=cix_yolov8s`，原 CPU 设置保存到 `person_detection_cpu_fallback`。样例人数一致仍不代表完成实际场景准确率验证。

报告位置：`data/person_npu_comparison.html`、`data/person_npu_comparison.json`；只有实际通过才生成验证成功标记。页面中显示的模型来源、设备和事件记录应与实际后端一致。

切换后，在网页原终端 Ctrl+C，重新执行：

```bash
./start-web.sh
```

浏览器 Ctrl+F5 刷新。上传图片、Agent 和本地视频持续监测都支持该人员检测后端。持续监测按每个模型配置的 Python 分别启动常驻进程：CIX 人员检测使用 `modelscope-venv` 的 NOE 环境，烟火使用 `.venv-fire` 的 CPU 环境。停止时优先通知子进程释放模型，超时再结束进程。

如普通终端也没有 `/dev/aipu`，先核对板卡驱动与当前运行环境。不要在没有设备的情况下把配置改为已验证。项目中的比较脚本会报告缺少设备。当前 Codex 沙箱看不到该设备，未完成真实 NPU 实测，也未自动切换配置。

## 烟火检测如何变成 `.cix`

可以先导出现有 DAMO 烟火模型到 ONNX，检查 CIX 编译兼容性；也可以训练/微调一个小型 YOLOv8n/s 烟火模型。采用后者时可参考已有 CIX YOLO 工程结构，但训练后的输出类别和预处理必须重新适配。

推荐工作流程：

1. 准备能识别烟雾与明火的权重。已有 VOC 数据集可用于训练，但训练、验证、测试应按场景或视频来源分开，避免相邻画面泄漏。普通 COCO YOLO 权重需要相应训练。
2. 导出固定输入大小、batch=1 的 ONNX。先验证 PyTorch 与 ONNX 对相同图片的框、分数和类别；尽量把不受编译器支持的 NMS/后处理移到 CPU。DAMO 模型须处理其背景类别映射。
3. 在 x86 Linux 主机建立 Python 3.10 编译环境，安装本地 CixBuilder wheel。使用烟火与普通场景图片制作校准数据，保持颜色顺序、归一化和缩放方式与训练、导出及运行完全一致。
4. 根据实际 ONNX 输入输出名称编写 cfg，执行 `cixbuild <实际配置文件>`。本机官方 YOLOv8s 配置可参考：`/home/orangepi/.cache/modelscope/models/cix--ai_model_hub_26_Q2/snapshots/master/models/ComputeVision/Object_Detection/onnx_yolov8_s/cfg/yolov8_sbuild.cfg`。它的 `trigger_float_op` 含模型层编号，不能直接复用到新模型。
5. 把编译结果与类别表拷回板卡，通过 NOE 运行。对比 ONNX 与 NPU 的框、类别和分数，测量漏检、误报与连续运行的速度，再接入 smoke/fire 提醒。

当前人员适配器的 BGR 拉伸预处理只对应已检查的官方模型；自训 YOLO 可能要求 RGB 与 letterbox，不能盲目套用。人员适配器还固定了 80 类和 person=0，不能直接把新的两类烟火模型路径填进去。

## 2026-11-24：现有 DAMO 权重已完成 CIX 编译

本次重新检查发现 `/sys/module/aipu/initstate` 为 `live`，`/sys/class/misc/aipu/dev` 为 `10:120`，系统还安装了 cix-npu-driver-dkms、cix-noe-umd 和 cix-ai-engine。**当前执行环境看不到 `/dev/aipu` 不等于板卡没有驱动**；先在普通桌面终端运行 `ls -l /dev/aipu`。不要仅因沙箱看不到设备就重装驱动，也不要人工创建节点绕过执行环境限制。

本地 SDK 手册第 3.3.1 节还支持 PyTorch 计算图，官方 `torch_centernet_resnet50` 示例使用 `torch.jit.load` 和 `model_type = torch`。因此这次先导出固定输入 TorchScript，避免依赖当前未安装的 ONNX 导出包。原始 DAMO `.pt` 是权重文件；新生成的 `.pt` 是包含计算图的 TorchScript，二者不能仅凭扩展名混用。

已在本机生成：

| 编译结果 | 图文件 | 原始输出形状 | 类别顺序 | 一致性验证 | 起步校准数据 |
| --- | --- | --- | --- | --- | --- |
| `data/cix-compiled-20261124/person/` | `damo_person.cix` | `[1,8400,5]` | person | 编译前 6 张独立于校准数据的样例做过 TorchScript 对照 | 现有游客实拍视频抽取的 5 帧 |
| `data/cix-compiled-20261124/fire_smoke/` | `damo_fire_smoke.cix` | `[1,8400,7]` | background、smoke、fire | 编译前 4 张独立于校准数据的样例做过 TorchScript 对照 | 已有烟火数据 train_list 中均匀抽取 32 张 |

每个目录包含 `build.cfg`、`datasets/calibration_data.npy`、`manifest.json` 和操作说明。manifest 记录图文件、权重及校准图片的 SHA256、实际输入输出名称、检查结果与未验证事项。校准图片与一致性验证图片按内容 SHA256 区分。

这里验证的是 **PyTorch 与 TorchScript 在相同固定输入下的原始张量一致性**。之后两份 TorchScript 都由 CixBuilder 编译成功；日志分别位于 `person/build-ubuntu22.log` 和 `fire_smoke/build-ubuntu22-retry.log`，压缩包内 `compiled-sha256.txt` 校验通过。原压缩包内的 manifest 保留了编译前的 `cix_compiled: false`；解压目录的 manifest 已按成功日志更正，原压缩包保持不变。真实 NOE 运行、量化后的检测精度和板端 NPU 推理仍未验证。固定方形补零也与当前网页的可变尺寸补零有所不同，未来须比较检测框和误报、漏报。5 帧同一游客视频和 32 张训练图片只够用作编译流程起步，量产前还应扩大到覆盖目标场景、光照和目标尺度的校准集，并保留独立测试集。

**不能直接把这两份 `.cix` 覆盖现用 `.pt` 或改配置路径来完成切换。** 当前 `detect_person.py` 和 `detect_fire_smoke.py` 调用 ModelScope DAMO CPU；唯一已有的 CIX YOLO 适配器固定为 COCO 输出 `[1,84,8400]`，而这两份 DAMO 图分别输出 `[1,8400,5]` 和 `[1,8400,7]`，预处理及解码也不同。接入前需要写 DAMO NOE 运行适配器、匹配后处理和配置分支，再在有 `/dev/aipu` 的板卡上对照 CPU 结果验收。当前执行沙箱没有 `/dev/aipu`，因此本次不能完成真实 NPU 推理或激活配置。

这两套 DAMO 候选的预处理为 RGB、float32、0–255、不放大小图、最长边缩到 640、右侧和底部补零至 `[1,3,640,640]`。输出最后一维为 `x1,y1,x2,y2` 加各类分数。后处理在 CPU 上做分类型阈值和 NMS，过滤 background，再恢复原图坐标。**不能使用现有 COCO YOLOv8 适配器的 BGR、除以 255 和固定 80 类解码。**

### 在板卡重新导出

导出工具是 `tools/export_damo_cix.py`。输出目录已存在 manifest 时会拒绝覆盖，重测请指定新目录：

```bash
cd /home/orangepi/emergency-agent
.venv-fire/bin/python tools/export_damo_cix.py \
  --model fire_smoke \
  --calibration-list /home/orangepi/Downloads/fire_smoke/fire_smoke/train_list.txt \
  --calibration-limit 32 \
  --output data/cix-export/fire_smoke-new
```

人员候选用 `--model person`，校准列表可以填写当前已准备的 `data/cix-export/calibration/person-list.txt`。当前配置仍然运行原 CPU 模型，导出不切换网页后端。

### 在另一台 x86_64 Linux 电脑上重新编译

现有两份模型已经编译完成；只有需要重编候选图时才需要下面的环境。CixBuilder wheel 的元数据要求 Python `>=3.10,<3.11`，平台是 `linux_x86_64`；SDK 手册列出 Ubuntu 20.04/22.04。电脑需要安装 Python 3.10 环境和 wheel 的依赖。普通 ARM Docker 容器不会自动变成此编译器所需的 x86 主机。

把板卡上的候选包和这个安装包拷到电脑：

`/home/orangepi/Downloads/cix_noe_sdk_26_q2_release/noe_sdk_26_q2_release/cixbuilder-6.1.3753.3-cp310-none-linux_x86_64.whl`

以下命令中的目录表示电脑上实际保存的位置：

```bash
python3.10 -m venv ~/cix-builder-venv
source ~/cix-builder-venv/bin/activate
pip install ./cixbuilder-6.1.3753.3-cp310-none-linux_x86_64.whl
cixbuild -v
cd cix-export/fire_smoke
set -o pipefail
cixbuild build.cfg 2>&1 | tee build.log
```

人员模型编译时进入 `cix-export/person`，同样执行 `cixbuild build.cfg`。配置的目标为 `X2_1204MP3`。本机这两份配置都已实际编译成功，日志末尾为 `build success`；压缩包内的 `.cix` 哈希已核验。若今后重编时解析器对算子或量化报错，应保留 build.log 并按日志适配。编译成功只证明产物生成，不等于 NOE 运行或检测精度通过。

已有官方 `yolov8s.cix` 可以继续诊断，接入成本较低，但目前尚未通过人员检测验收。重新编译 DAMO 人员模型可保留现有架构；这条路线以及烟火模型编译都不需要等待 YOLOv8s 验收通过。

### NPU 接入与验收还需完成

1. 在能访问 `/dev/aipu` 的板卡终端，用 NOE 分别加载 `data/cix-compiled-20261124/person/damo_person.cix` 和 `data/cix-compiled-20261124/fire_smoke/damo_fire_smoke.cix` 做实机推理。
2. 为 DAMO 原始张量实现匹配的 NPU 后处理；现有 `tools/detect_yolov8_npu.py` 不能直接替换模型路径来适配它。
3. 对比同样预处理下 TorchScript / 量化输出 / NPU 的框、类别与分数，检查精度下降；独立测试集另测召回、误报和漏报。
4. 实测 NPU forward 时间、预处理加后处理总时间、持续监测间隔、内存与长时间稳定性。
5. 通过后再接入人员、烟火调用路径、持续监测和网页能力状态；保留 CPU 回退配置。

## Qwen 的处理

当前 Qwen 是 GGUF 模型。下载在 CIX 模型库中的 Qwen 文档，不意味着模型已转换成 `.cix` 或支持当前 NPU 推理接口。本机官方文档 `.../cix--ai_model_hub_26_Q2/snapshots/master/models/Generative_AI/LLM/Qwen2.5-3B-Instruct.md` 提供的是 llama.cpp CPU/Vulkan 与 MNN CPU/OpenCL 示例。

持续监测报警链路不调用 Qwen，因此先让视觉检测运行到 NPU 更直接。Qwen 如需加速，可单独检查本机 CIX llama.cpp Vulkan 构建与显存、生成速度，不能把当前 GGUF 简单改后缀为 `.cix`。
