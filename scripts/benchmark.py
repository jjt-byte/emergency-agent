#!/usr/bin/env python3
"""Re-run bounded CPU inference measurements, preserving model/image hashes."""
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['person', 'fire_smoke'])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT/'data/benchmark-current.json')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error('repeats must be 1..100')
    config = json.loads((ROOT/'config.json').read_text())
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {int(c) for c in config['cpu_affinity'].split(',')})
    if args.worker:
        settings = config['person_detection' if args.worker == 'person' else 'fire_smoke']
        image = ROOT/('samples/000000252219.jpg' if args.worker == 'person' else 'samples/fire-smoke/ESE_fire_10.jpg')
        from tools.detect_fire_smoke import digest
        start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            if args.worker == 'person':
                from tools.detect_person import LocalDetector
            else:
                from tools.detect_fire_smoke import LocalDetector
            detector = LocalDetector(settings['model_dir'], settings['threshold'], settings['max_side'], settings['threads'])
        load = time.perf_counter()-start
        first_start = time.perf_counter()
        first = detector.infer(image)
        first_wall = time.perf_counter()-first_start
        samples, internal, labels = [], [], []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = detector.infer(image)
            samples.append(time.perf_counter()-start)
            internal.append(result['elapsed_seconds'])
            labels.append([d['label'] for d in result['detections']])
        print(json.dumps({'kind':args.worker, 'image':str(image.relative_to(ROOT)),
            'image_sha256':digest(image), 'model_sha256':first['model_sha256'],
            'load_seconds':load, 'first_infer_wall_seconds':first_wall,
            'warm_samples_wall_seconds':samples, 'warm_samples_internal_seconds':internal,
            'warm_mean_seconds':statistics.mean(samples), 'warm_median_seconds':statistics.median(samples),
            'warm_min_seconds':min(samples), 'warm_max_seconds':max(samples),
            'labels_per_repeat':labels, 'threads':settings['threads'], 'device':'CPU'}))
        return
    results = []
    for kind in ('person', 'fire_smoke'):
        before = time.perf_counter()
        run = subprocess.run([config['person_detection']['python'],str(Path(__file__).resolve()),
            '--worker',kind,'--repeats',str(args.repeats)],capture_output=True,text=True,timeout=240)
        if run.returncode:
            raise RuntimeError(run.stderr[-3000:])
        record=json.loads(run.stdout)
        record['whole_worker_seconds']=time.perf_counter()-before
        results.append(record)
    result={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'architecture':platform.machine(),'python':platform.python_version(),
        'cpu_affinity':sorted(os.sched_getaffinity(0)), 'repeats':args.repeats,
        'scope':'one image per model; one first inference plus repeated warm calls; no end-to-end video/agent benchmark',
        'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
