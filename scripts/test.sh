#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
task_python="${TEST_PYTHON:-$task_root/.venv/bin/python}"
[[ -x "$task_python" ]] || task_python=python3
if [[ ! -f config.json ]]; then
  python3 scripts/configure.py --runtime-python "$task_python"
fi
"$task_python" -m unittest discover -s tests -p 'test_*.py'
for task_test in tests/test_ui*.js; do
  node "$task_test"
done
