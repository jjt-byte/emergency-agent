#!/usr/bin/env bash
set -eu
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$task_root"
exec python3 scripts/run.py webapp.py "$@"
