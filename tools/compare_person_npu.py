#!/usr/bin/env python3
"""Compare actual DAMO CPU / CIX YOLOv8s NPU results; activate only after sample checks."""
import argparse
import base64
import datetime
import html
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.detect_yolov8_npu import DEFAULT_MODEL, fingerprint, native_logs_to_stderr

REPORT = ROOT/'data/person_npu_comparison.json'
MARKER = ROOT/'data/person_npu_validation.json'
CANDIDATE = {'backend': 'cix_yolov8s', 'model_path': str(DEFAULT_MODEL),
             'python': os.environ.get('NPU_PYTHON', sys.executable), 'threshold': .5,
             'nms_threshold': .45, 'validation_file': str(MARKER)}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def cases():
    # Legacy foreground counts; one scene also contains distant background people.
    old = json.loads((ROOT/'data/person_detection_validation.json').read_text())['results']
    counts = {'six_person_user_image': 6, 'previous_three_person_image': 3,
              '000000006954': 5, '000000013291': 4, '000000081988': 5, 'blank_negative': 0}
    return [{'name': row['name'], 'image': row['image'], 'expected_count': counts[row['name']],
             'reference_complete': row['name'] != '000000013291',
             'reference_note': ('原参考仅统计四名前景人员，背景还有小尺度人员，不能直接作全图人数真值'
                                if row['name'] == '000000013291' else '人工样例人数，未提供完整参考框')}
            for row in old if row['name'] in counts]


def worker(backend, output, repeats):
    config = json.loads((ROOT/'config.json').read_text())
    baseline = config.get('person_detection_cpu_fallback', config['person_detection'])
    if baseline.get('backend') != 'damoyolo':
        raise ValueError('找不到 DAMO CPU 对比配置')
    if hasattr(os, 'sched_setaffinity'):
        desired = {int(x) for x in config.get('cpu_affinity', '0,1,6-11').replace('6-11','6,7,8,9,10,11').split(',')}
        allowed = os.sched_getaffinity(0) & desired
        if allowed:
            os.sched_setaffinity(0, allowed)
    started = time.perf_counter()
    with native_logs_to_stderr():
        if backend == 'damo':
            from tools.detect_person import LocalDetector
            detector = LocalDetector(baseline['model_dir'], .5, baseline.get('max_side', 640), baseline.get('threads', 8))
        else:
            from tools.detect_yolov8_npu import LocalDetector
            detector = LocalDetector(CANDIDATE['model_path'], .5, .45)
    load_seconds = time.perf_counter()-started
    rows = []
    try:
        for case in cases():
            detector.infer(case['image'])  # Untimed warmup for each image shape.
            samples = [detector.infer(case['image']) for _ in range(repeats)]
            counts = [len(s['detections']) for s in samples]
            rows.append(dict(case, result=samples[-1], count=counts[-1], stable=len(set(counts)) == 1,
                             median_seconds=statistics.median(s['elapsed_seconds'] for s in samples),
                             samples_seconds=[s['elapsed_seconds'] for s in samples]))
            print(f"{backend}: {case['name']} 检出 {counts[-1]} 人 / 参考 {case['expected_count']} 人", flush=True)
    finally:
        if backend == 'yolo':
            detector.close()
    write_json(output, {'backend': backend, 'load_seconds': load_seconds, 'results': rows,
                        'repeats': repeats, 'warmups_per_image': 1})


def render_report(report):
    parts = ['<!doctype html><meta charset="utf-8"><title>人员模型实测对比</title>',
             '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}svg{width:100%;max-height:480px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}</style>',
             '<h1>人员检测：DAMO CPU 与 CIX YOLOv8s NPU</h1>',
             '<p>六张样例的人工参考人数对比；未提供参考框，不能据此计算召回率或认定没有漏检。耗时为模型加载后的图像预处理、推理、后处理总耗时中位数，不含 Qwen 和网页排队。</p>',
             '<p>验证状态：'+html.escape(report['status'])+'</p>',
             '<table><tr><th>图片</th><th>参考人数</th><th>DAMO人数 / 秒</th><th>NPU人数 / 秒</th></tr>']
    for d, y in zip(report['damo']['results'], report['yolo']['results']):
        parts.append(f"<tr><td>{html.escape(d['name'])}</td><td>{d['expected_count']}</td><td>{d['count']} / {d['median_seconds']:.4f}</td><td>{y['count']} / {y['median_seconds']:.4f}</td></tr>")
    parts.append('</table>')
    for warning in report.get('reference_warnings', []):
        parts.append('<p>参考标注待复核：'+html.escape(warning['name']+'：'+warning['note'])+'</p>')
    for d, y in zip(report['damo']['results'], report['yolo']['results']):
        parts.append('<h2>'+html.escape(d['name'])+'</h2><div class="pair">')
        raw = base64.b64encode(Path(d['image']).read_bytes()).decode()
        mime = 'image/png' if Path(d['image']).suffix.lower() == '.png' else 'image/jpeg'
        for name, row, color in [('DAMO CPU', d, '#00cfff'), ('YOLOv8s NPU', y, '#ff9f00')]:
            w, h = row['result']['image_size']
            parts.append(f'<div><h3>{name}：{row["count"]} 人</h3><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><image width="{w}" height="{h}" href="data:{mime};base64,{raw}"/>')
            for i, det in enumerate(row['result']['detections'], 1):
                x1,y1,x2,y2 = det['box']
                parts.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{color}" stroke-width="3"/><text x="{x1}" y="{max(20,y1)}" fill="{color}" stroke="black" stroke-width=".4" font-size="22">#{i} {det["confidence"]:.2f}</text>')
            parts.append('</svg></div>')
        parts.append('</div>')
    REPORT.with_suffix('.html').write_text(''.join(parts))


def compare(activate=False, repeats=3):
    config = json.loads((ROOT/'config.json').read_text())
    baseline = config.get('person_detection_cpu_fallback', config['person_detection'])
    report = {'status': 'running', 'created_at': datetime.datetime.now().isoformat(),
              'candidate': CANDIDATE, 'scope': 'six visually counted samples, no bbox ground truth',
              'activated': False}
    write_json(REPORT, report)
    if not Path('/dev/aipu').exists():
        report.update(status='blocked', message='当前执行环境看不到 /dev/aipu，尚未运行 NPU 对比；请在板卡本机终端运行 ./test-person-npu.sh --activate')
        write_json(REPORT, report)
        print(report['message'], file=sys.stderr)
        return 2
    try:
        for backend, python in [('damo', baseline['python']), ('yolo', CANDIDATE['python'])]:
            out = ROOT/f'data/person_npu_{backend}_measurements.json'
            result = subprocess.run([python, str(Path(__file__).resolve()), '--worker', backend,
                                     '--output', str(out), '--repeats', str(repeats)],
                                    cwd=ROOT, text=True, capture_output=True, timeout=600)
            print(result.stdout, end='', flush=True)
            if result.returncode:
                raise RuntimeError(f'{backend} 推理未成功：'+result.stderr[-2500:])
            report[backend] = json.loads(out.read_text())
        yrows, drows = report['yolo']['results'], report['damo']['results']
        report['reference_warnings'] = [{'name': row.get('name', ''),
                                         'note': row.get('reference_note', '人数参考不完整')}
                                        for row in yrows if row.get('reference_complete') is False]
        passed = (len(yrows) == len(drows) == 6
                  and not report['reference_warnings']
                  and all(y['stable'] and y['count'] == y['expected_count'] for y in yrows))
        report['status'] = 'passed' if passed else 'needs_review'
        report['mean_image_seconds'] = {key: statistics.mean(row['median_seconds'] for row in report[key]['results'])
                                        for key in ('damo','yolo')}
        report['speedup'] = report['mean_image_seconds']['damo']/max(report['mean_image_seconds']['yolo'], 1e-9)
        if passed:
            write_json(MARKER, {'status':'passed', 'fingerprint':fingerprint(CANDIDATE),
                                'comparison':str(REPORT), 'scope':report['scope']})
        else:
            write_json(MARKER, {'status':'needs_review', 'message':
                '参考标注不完整，须先复核背景人员；未自动切换' if report['reference_warnings']
                else '样例人数不符或多次结果不稳定，未自动切换'})
        if passed and activate:
            current = json.loads((ROOT/'config.json').read_text())
            if current != config:
                raise RuntimeError('测试期间配置发生变化，请重新测试后切换')
            backup = ROOT/'data/backups'/('person-config-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'.json')
            write_json(backup, config)
            config.setdefault('person_detection_cpu_fallback', baseline)
            config['person_detection'] = CANDIDATE
            write_json(ROOT/'config.json', config)
            report.update(activated=True, config_backup=str(backup))
        write_json(REPORT, report)
        render_report(report)
        print(json.dumps({'status':report['status'], 'activated':report['activated'],
                          'speedup':round(report['speedup'],3), 'report':str(REPORT.with_suffix('.html'))}, ensure_ascii=False))
        if report['activated']:
            print('已启用 YOLOv8s NPU。请在原网页终端 Ctrl+C，再运行 ./start-web.sh --host 0.0.0.0 --port 8080，浏览器 Ctrl+F5 刷新。')
        return 0 if passed else 3
    except Exception as exc:
        report.update(status='failed', message=str(exc))
        write_json(REPORT, report)
        print(str(exc), file=sys.stderr)
        return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--activate', action='store_true')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--worker', choices=['damo','yolo'], help=argparse.SUPPRESS)
    p.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 1 <= args.repeats <= 10:
        p.error('repeats 必须为 1..10')
    if args.worker:
        worker(args.worker, args.output, args.repeats)
        return 0
    return compare(args.activate, args.repeats)


if __name__ == '__main__':
    sys.exit(main())
