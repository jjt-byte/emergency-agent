#!/usr/bin/env bash
set -eu
task_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$task_root"
exec /usr/bin/python3 "$task_root/desktop_app.py" "$@"
