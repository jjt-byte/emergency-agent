#!/usr/bin/env python3
"""Generate machine-local absolute configuration from the committed template."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def configure(runtime_python, llama_cli=None):
    config = json.loads((ROOT / 'config.example.json').read_text())
    for key in ('model_path', 'vision_model', 'llama_cli'):
        config[key] = str(ROOT / config[key])
    config['vision_python'] = str(Path(runtime_python).absolute())
    config['runtime_python'] = config['vision_python']
    if llama_cli:
        config['llama_cli'] = str(Path(llama_cli).resolve())
    # Intersect the original big-core preference with CPUs available on this host.
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    preferred = [c for c in (0, 1, 6, 7, 8, 9, 10, 11) if c in allowed]
    cpus = preferred or allowed[:8]
    config['cpu_affinity'] = ','.join(map(str, cpus))
    config['threads'] = min(8, len(cpus))
    for section in ('fire_smoke', 'person_detection'):
        settings = config[section]
        settings['model_dir'] = str(ROOT / settings['model_dir'])
        settings['validation_file'] = str(ROOT / settings['validation_file'])
        settings['python'] = str(Path(runtime_python).absolute())
        settings['threads'] = config['threads']
    (ROOT / 'data').mkdir(exist_ok=True)
    (ROOT / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-python', default=sys.executable)
    parser.add_argument('--llama-cli')
    args = parser.parse_args()
    configure(args.runtime_python, args.llama_cli)
    print(ROOT / 'config.json')
