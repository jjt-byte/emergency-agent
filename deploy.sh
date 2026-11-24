#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$task_root"
task_runtime=""
task_llama=""
task_models=""
task_samples=""
task_wheels=""
task_no_start=0
task_host=127.0.0.1
task_port=8080
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) task_models="$2"; shift 2;;
    --samples) task_samples="$2"; shift 2;;
    --runtime-python) task_runtime="$2"; shift 2;;
    --llama-cli) task_llama="$2"; shift 2;;
    --wheel-dir) task_wheels="$2"; shift 2;;
    --no-start) task_no_start=1; shift;;
    --host) task_host="$2"; shift 2;;
    --port) task_port="$2"; shift 2;;
    -h|--help)
      cat <<'USAGE'
Usage: bash deploy.sh [--models /path/models.tar.gz] [--samples /path/local-samples.tar.gz]
  [--runtime-python /path/python] [--llama-cli /path/llama-completion]
  [--wheel-dir /path/wheels] [--host 127.0.0.1] [--port 8080] [--no-start]
Fresh install: creates .venv, installs requirements, builds pinned CPU llama.cpp if needed.
Offline install: pass runtime-python (or complete wheel-dir) AND an existing llama-cli.
Real inference validation requires the separate local sample bundle.
USAGE
      exit 0;;
    *) echo "未知参数: $1" >&2; exit 2;;
  esac
done
for task_binary in python3 ffmpeg ffprobe taskset; do
  command -v "$task_binary" >/dev/null || { echo "缺少系统依赖: $task_binary，见 README" >&2; exit 1; }
done
if [[ -n "$task_models" ]]; then
  python3 scripts/assets.py models --archive "$task_models"
else
  python3 scripts/assets.py models
fi
if [[ -n "$task_samples" ]]; then
  python3 scripts/assets.py samples --archive "$task_samples"
else
  python3 scripts/assets.py samples
fi
if [[ -z "$task_runtime" ]]; then
  [[ -x .venv/bin/python ]] || python3 -m venv .venv
  task_runtime="$task_root/.venv/bin/python"
  task_pip_args=()
  if [[ -n "$task_wheels" ]]; then
    task_pip_args+=(--no-index --find-links "$task_wheels")
  fi
  "$task_runtime" -m pip install -r requirements.txt "${task_pip_args[@]}"
fi
if [[ -z "$task_llama" ]]; then
  task_llama="$task_root/vendor/llama.cpp/build-cpu/bin/llama-completion"
  [[ -x "$task_llama" ]] || bash scripts/build_llama.sh
fi
[[ -x "$task_llama" ]] || { echo "llama-completion 不可执行: $task_llama" >&2; exit 1; }
python3 scripts/configure.py --runtime-python "$task_runtime" --llama-cli "$task_llama"
"$task_runtime" scripts/validate_models.py
python3 tools/init_knowledge.py
python3 scripts/doctor.py
echo "部署完成。启动命令: bash start-web.sh --host $task_host --port $task_port"
if [[ "$task_no_start" == 0 ]]; then
  exec bash start-web.sh --host "$task_host" --port "$task_port"
fi
