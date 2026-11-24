# 应急先行 · 离线应急巡查 Agent

面向 Orange Pi 6 Plus / CIX P1 的端侧应急辅助原型。通过本地网页上传图片、导入视频、进行中文问答，完成人员与烟火检测、事件记录和报告查询。依赖与资源准备完成后，推理在本地执行。

默认使用 **Qwen2.5-3B-Instruct Q4_K_M + llama.cpp CPU**、**DAMO-YOLO 人员 CPU** 和 **DAMO-YOLO 烟火 CPU**。项目提供可选 CIX NPU 人员检测适配与诊断工具，启用状态由板端验证决定。

## 功能

- 图片问答、多轮场景追问、异步分析任务与取消。
- 受限工具 Agent，最多三轮工具规划，结合检测结果、知识检索与证据约束生成回答。
- 本地视频定时抽帧、检测进程常驻、重复事件合并、连续检出升级、事件截图与浏览器声音提醒。
- 图片和模型哈希留存，生成 JSON、Markdown、HTML 报告，支持人工确认与历史查询。
- SQLite 离线知识检索、批量评估、可选 GTK/WebKit 桌面壳。

当前支持本地图片与视频文件。摄像头、RTSP、实体报警器及远程通知尚待开发。知识资料由项目整理，待专业审核。检测失败显示未知，未检出仍需人工复核。

## 运行环境

原项目历史验证平台为 Orange Pi 6 Plus、CIX P1 CD8180、ARM64、约 31 GiB 内存、Debian 12、Linux 6.6.89-cix、Python 3.11.2。

| 组件 | 配置 |
|---|---|
| PyTorch / torchvision | 2.5.1 / 0.20.1，CPU |
| NumPy / Pillow | 1.26.4 / 11.3.0 |
| ModelScope / timm | 1.40.2 / 0.9.16 |
| llama.cpp 固定提交 | `d88040c704c8855c497d46c20675a3d492f70f94` |
| 本地语言模型 | Qwen2.5-3B-Instruct，Q4_K_M |
| 检测预处理 | RGB 0–255，最长边 640，stride 32 补零 |
| 默认阈值 | 人员 0.50，烟火 0.60 |
| 推理线程 | 最多 8，按允许的 CPU 集合配置 |

依赖版本见 [requirements.txt](requirements.txt)。原机完整环境快照供审计参考。其他架构、Windows 原生运行及不同 Python 版本的推理链路仍需验证。监测与任务管理依赖 Linux 进程接口。

## 快速部署

在 Debian / Ubuntu 目标板上进入仓库根目录。

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-dev python3-pip \
  build-essential cmake git ffmpeg util-linux libopenblas-dev libgomp1
```

准备模型包与本地验证样例包，再执行部署。资源说明、来源和自行准备步骤见 [资源准备指南](docs/资源准备指南.md)。

```bash
bash deploy.sh \
  --models /absolute/path/emergency-agent-models-20261124.tar.gz \
  --samples /absolute/path/emergency-agent-local-samples-20261124.tar.gz
```

部署脚本校验资源、建立 `.venv`、安装依赖、构建固定提交的 llama.cpp、生成 `config.json`、执行真实推理冒烟验证、初始化知识库并启动网页。首次安装需要访问依赖源。建议预留 15 GB 磁盘。

浏览器访问 `http://127.0.0.1:8080`。后续启动可使用以下命令。

```bash
bash start-web.sh
# 受信任局域网演示可指定监听地址与端口
bash start-web.sh --host 0.0.0.0 --port 8082
```

局域网访问地址为 `http://板卡IP:8082`。网页服务器当前采用无账号认证的开发服务器，公网部署需要另行配置认证与反向代理。

### 复用现有运行环境

```bash
bash deploy.sh \
  --models /absolute/path/emergency-agent-models-20261124.tar.gz \
  --samples /absolute/path/emergency-agent-local-samples-20261124.tar.gz \
  --runtime-python /absolute/path/environment/bin/python \
  --llama-cli /absolute/path/llama.cpp/build-cpu/bin/llama-completion \
  --no-start
bash start-web.sh --port 8082
```

网页、命令行和桌面后端使用生成配置中的 `runtime_python`。旧配置兼容 `vision_python`，临时覆盖可设置 `AGENT_PYTHON=/absolute/path/python`。桌面壳自身由系统 Python 加载 GTK，后端沿用配置的推理环境。

完全离线安装可通过 `--wheel-dir /absolute/path/wheels` 提供适配 ARM64/Python 3.11 的完整 wheel 集合，同时通过 `--llama-cli` 提供目标系统上构建的程序及其共享库。

### 验证与其他入口

以下 `.venv/bin/python` 可替换为部署时指定的外部解释器。

```bash
.venv/bin/python scripts/doctor.py
.venv/bin/python scripts/validate_models.py
.venv/bin/python scripts/benchmark.py --repeats 5
bash start.sh
```

`validate_models.py` 使用两张人员样例与四张烟火样例生成当前配置的验证记录。烟火样例存在已知漏检，冒烟验证结果需结合原始输出复核。

照片序列演示视频可通过以下命令生成，然后在网页“持续监测”中导入。生成前需导入样例包并安装 FFmpeg。

```bash
.venv/bin/python tools/make_monitor_demo.py
.venv/bin/python tools/make_monitor_demo.py --suite
```

浏览器声音需要先在页面启用。照片序列用于演示抽帧、事件和留痕流程。桌面组件配置见 [桌面版使用说明](docs/桌面版使用说明.md)。

## 软件测试

软件测试依赖 NumPy、Pillow 和 Node.js，可在 Linux 上独立执行。

```bash
python3 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
TEST_PYTHON="$PWD/.venv-test/bin/python" bash scripts/test.sh
```

GitHub Actions 配置为 Ubuntu、Python 3.11、Node.js 20。覆盖资源导入校验、Agent 路由、证据约束、报告、任务生命周期、持续监测、API 边界及启动解释器选择。

[原交付验收记录](benchmarks/release-verification.json) 保存历史 117 项 Python 测试与 6 组前端测试通过的结果。当前整理版本的本机验证范围见 [仓库整理与验证说明](docs/仓库整理与验证说明.md)。板端推理和云端 CI 结果以实际运行记录为准。

## 可选 NPU 与批量评估

- 人员 NPU 需要真实 `/dev/aipu`、匹配的 NOE Python SDK、CIX 运行库与模型。入口为 `test-person-npu.sh` 和 `diagnose-person-npu.sh`。历史候选状态为 `needs_review`。
- NPU 对比工具使用六张图的人工人数参考。新部署的冒烟验证只生成两张人员样例记录，完整对比需要另行准备六张图及本机参考路径。
- 烟火批量评估需要额外的 `fire_smoke.zip` 与 `fire_600.zip` 数据集，放入 `data/datasets/smoke_fire_datasets_1/`。来源与评估口径见 [资源准备指南](docs/资源准备指南.md)。

## 历史实测

| 项目 | 历史结果 | 测量范围 |
|---|---|---|
| LLM 生成速度 | 约 16 token/s | CPU，8 线程，tg128，两次测量 |
| 人员检测热调用 | 平均 0.243522 秒/图 | 单张图重复 5 次 |
| 烟火检测热调用 | 平均 0.248329 秒/图 | 单张图重复 5 次 |
| 烟火冒烟样例 | 3 张正例中 1 张检出 | 另有 1 张负例，存在漏检 |
| NPU 候选 | `needs_review` | 人数参考与检出仍待复核 |

完整计时、模型加载耗时、原始记录及来源见 [benchmarks](benchmarks/README.md)。语言模型、视觉检测和完整 Agent 的测量口径各见原始记录。技术方案详见 [技术白皮书](docs/技术白皮书.md) 与 [PDF](docs/技术白皮书.pdf)。

## 目录

```text
agent.py / agent_loop.py / triage.py   任务、工具规划、证据与规则
webapp.py / web/                      HTTP API 与网页
monitor.py / tools/monitor_video.py   视频巡查与检测进程
desktop_app.py / desktop_runtime.py   可选桌面壳
scripts/ / deploy.sh                  部署、启动、验证和复测
tools/                               检测、数据处理与 NPU 工具
knowledge/                           离线知识资料
tests/                               软件测试与合成空白图片
benchmarks/                          归档实测、来源与资源清单
docs/                                技术文档、历史记录与第三方声明
models/ / samples/                   外部资源导入位置
```

运行时生成的 `config.json`、`data/`、虚拟环境、权重与录像由 `.gitignore` 排除。历史文档保留原机路径和早期状态，部署操作以本 README 为准。

## 上传到 GitHub

### 网页上传

解压源码 ZIP，进入包含 `README.md` 的目录。在 GitHub 创建空仓库后，点击页面中的 `uploading an existing file`。

1. 先拖入 `docs` 和 `benchmarks` 两个文件夹，共 36 个文件，点击 `Commit changes`。
2. 再通过 `Add file → Upload files` 上传其余文件和文件夹，共 89 个文件，点击 `Commit changes`。

上传后核对仓库根目录包含 `agent.py`、`README.md`、`.github`、`tests`、`tools` 和 `web`。部署命令和 CI 显式使用 `bash` 调用 shell 脚本。

### 使用 Git 提交

在 GitHub 创建空仓库，进入本目录执行以下命令，并替换远程仓库地址。本地整理目录已初始化 `main` 分支，ZIP 解压副本需先执行 `git init -b main`。

```bash
git add .
git add --chmod=+x -- '*.sh'
git diff --cached --stat
git commit -m "Prepare offline emergency inspection agent"
git remote add origin https://github.com/YOUR_USERNAME/emergency-agent.git
git push -u origin main
```

首次提交如需身份配置，请使用自己的 `git config user.name` 与 `git config user.email`。上述第二条命令设置 shell 脚本的 Git 可执行位，适用于 Windows 解压副本。

## 许可

第一方代码沿用 [LICENSE](LICENSE) 中的版权保留声明，开源授权方案由项目持有人选定。第三方组件、模型与素材适用各自条款，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 `docs/licenses/`。模型、现场照片与运行历史通过独立资源管理。
