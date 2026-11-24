#!/usr/bin/env python3
"""Launch application entry points with the configured inference environment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def runtime_python(root=ROOT):
    root = Path(root)
    candidate = os.environ.get('AGENT_PYTHON')
    if not candidate:
        config_path = root / 'config.json'
        config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
        candidate = config.get('runtime_python') or config.get('vision_python')
    if not candidate:
        local = root / '.venv/bin/python'
        candidate = str(local) if local.is_file() else sys.executable
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return str(path)
    if path.parent != Path('.'):
        return str(root / path)
    return shutil.which(str(path)) or str(root / path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('entrypoint', choices=['webapp.py', 'agent.py'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    python = runtime_python()
    os.execv(python, [python, str(ROOT / args.entrypoint), *args.arguments])


if __name__ == '__main__':
    main()
