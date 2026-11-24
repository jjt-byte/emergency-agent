#!/usr/bin/env bash
set -eu
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$task_root"
if [ "$#" -eq 0 ]; then
    exec python3 scripts/run.py agent.py interactive
fi
if [ "$1" = "--demo" ]; then
    exec python3 scripts/run.py agent.py run --task '检查图片中的人员并记录巡查风险' \
        --image "$task_root/samples/000000559842.jpg" --npu-detect --llm --confirm-after
fi
exec python3 scripts/run.py agent.py "$@"
