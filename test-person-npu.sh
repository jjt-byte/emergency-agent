#!/usr/bin/env bash
set -eu
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$task_root"
exec "${NPU_PYTHON:-python3}" tools/compare_person_npu.py "$@"
