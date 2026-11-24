# 模型资源

权重不提交 Git。配套 `emergency-agent-models-20261124.tar.gz` 中包含本项目原版实际使用的 Qwen GGUF、两个人员/烟火 CPU 模型，以及原版可选 CIX 检测器。文件列表、字节数、SHA256 见 `benchmarks/models_manifest.json`。

运行 `python3 scripts/assets.py models --archive /绝对路径/资源包.tar.gz` 导入。来源与许可证见 `THIRD_PARTY_NOTICES.md`。不要把 Qwen2.5-3B 当作 Apache-2.0 模型，也不要把整个资源包直接公开到 GitHub。
