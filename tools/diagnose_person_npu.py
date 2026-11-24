#!/usr/bin/env python3
"""Diagnose NPU input, suppression and timing; never change active configuration."""
import argparse
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from tools import detect_yolov8_npu as yolo
from tools import npu_image

HUB = ROOT/'models/person/cix-yolov8s'
PROFILES = ['legacy_ffmpeg', 'bgr_stretch', 'rgb_stretch', 'rgb_letterbox']


def references():
    from tools.compare_person_npu import cases
    return cases()


def threshold_matrix(raw, width, height, transform=None):
    return [{'threshold': score, 'nms_threshold': iou,
             'count': len(yolo.decode([raw], width, height, score, iou, transform=transform))}
            for score in [.25, .35, .5, .6] for iou in [.45, .6, .7]]


def original_findings():
    path = ROOT/'data/person_npu_comparison.json'
    if not path.is_file():
        return {'status': 'missing_previous_comparison'}
    previous = json.loads(path.read_text())
    rows = previous.get('yolo', {}).get('results', [])
    forwards = [r['result']['forward_seconds'] for r in rows if 'forward_seconds' in r.get('result', {})]
    return {'comparison_created_at': previous.get('created_at'),
            'status': previous.get('status'),
            'forward_mean_seconds': statistics.mean(forwards) if forwards else None,
            'total_mean_seconds': previous.get('mean_image_seconds', {}).get('yolo'),
            'damo_total_mean_seconds': previous.get('mean_image_seconds', {}).get('damo'),
            'observations': ['Six-person image has missed/merged people; inspect raw candidates and suppression.',
                             '000000013291 has four foreground people plus distant background people; four is not an exhaustive count.',
                             'Frame preprocessing spawned ffprobe and ffmpeg; forward time alone does not describe total latency.']}


def preprocessing_checks(cases, repeats):
    measurements = []
    for case in cases:
        row = {'name': case['name']}
        for name, prepare in [('legacy', yolo.prepare_image_ffmpeg), ('in_process', yolo.prepare_image)]:
            prepare(case['image'])
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                prepare(case['image'])
                samples.append(time.perf_counter()-started)
            row[name+'_median_seconds'] = statistics.median(samples)
        measurements.append(row)
    sample = HUB/'test_data/1.jpeg'
    binary = HUB/'datasets/input0.bin'
    similarity = None
    if sample.is_file() and binary.is_file():
        expected = np.fromfile(binary, np.uint8)
        if expected.size != 3*640*640:
            raise ValueError('官方输入校验文件尺寸不符')
        expected = expected.reshape(3, 640, 640).astype(np.float32)
        legacy, _ = yolo.prepare_image_ffmpeg(sample)
        candidate, _ = yolo.prepare_image(sample)
        similarity = {'reference': 'CIX bundled input0.bin vs test_data/1.jpeg',
                      'metric': 'mean absolute difference in uint8 pixel units; not model accuracy',
                      'legacy_bgr_mean_absolute_error': float(np.abs(legacy[0]*255-expected).mean()),
                      'candidate_bgr_mean_absolute_error': float(np.abs(candidate[0]*255-expected).mean()),
                      'candidate_rgb_mean_absolute_error': float(np.abs(candidate[0, ::-1]*255-expected).mean())}
    return {'implementation': npu_image.VERSION, 'repeats': repeats,
            'measurements': measurements, 'official_input_similarity': similarity}


def overlay(image, detections, target):
    from PIL import Image, ImageDraw
    with Image.open(image) as opened:
        picture = opened.convert('RGB')
    draw = ImageDraw.Draw(picture)
    for index, det in enumerate(detections, 1):
        draw.rectangle(det['box'], outline='#db4a28', width=3)
        draw.text((det['box'][0], max(0, det['box'][1]-14)),
                  f"#{index} {det['confidence']:.3f}", fill='black', stroke_width=1, stroke_fill='white')
    picture.save(target)


def hardware_checks(cases, directory, repeats, rows=None):
    detector = yolo.LocalDetector()
    if rows is None:
        rows = []
    try:
        for case in cases:
            for profile in PROFILES:
                print(f"正在检查 {case['name']} / {profile} …", flush=True)
                detector.infer_raw(case['image'], profile)  # Untimed warmup.
                samples = []
                counts = []
                for _ in range(repeats):
                    started = time.perf_counter()
                    raw, details = detector.infer_raw(case['image'], profile)
                    detections = yolo.decode([raw], *details['image_size'], transform=details['transform'])
                    samples.append({'total_seconds': time.perf_counter()-started,
                                    'preprocess_seconds': details['preprocess_seconds'],
                                    'forward_seconds': details['forward_seconds']})
                    counts.append(len(detections))
                stem = case['name']+'-'+profile
                np.savez_compressed(directory/(stem+'.npz'), output=raw)
                overlay(case['image'], detections, directory/(stem+'.jpg'))
                pred = yolo.prediction_rows([raw])
                before_nms = pred[(pred[:, 4:].argmax(axis=1) == 0) & (pred[:, 4] >= .05)]
                before_nms = before_nms[np.argsort(-before_nms[:, 4])][:100]
                rows.append({'name': case['name'], 'profile': profile,
                             'reference_foreground_or_total_count': case['expected_count'],
                             'reference_complete': case.get('reference_complete', True),
                             'reference_note': case.get('reference_note'),
                             'count': len(detections), 'sample_counts': counts,
                             'stable': len(set(counts)) == 1,
                             'timings': {key: statistics.median(s[key] for s in samples) for key in samples[0]},
                             'detections': detections, 'input': details,
                             'threshold_matrix': threshold_matrix(raw, *details['image_size'], transform=details['transform']),
                             'raw_candidate_xywh_person_score': before_nms[:, :5].tolist(),
                             'raw_output_file': stem+'.npz', 'overlay_file': stem+'.jpg'})
                # Preserve completed cases even if a later image/driver call fails.
                (directory/(stem+'.json')).write_text(json.dumps(rows[-1], ensure_ascii=False, indent=2)+'\n')
                print(f"{case['name']} / {profile}: {len(detections)} 人 / 总耗时 {rows[-1]['timings']['total_seconds']:.4f}s / 前向 {rows[-1]['timings']['forward_seconds']:.4f}s", flush=True)
    finally:
        detector.close()
    return rows


def render_report(report, directory):
    parts = ['<!doctype html><meta charset="utf-8"><title>人员 NPU 排查</title>',
             '<style>body{font:16px/1.8 sans-serif;max-width:1200px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:9px;text-align:left}img{max-width:100%}details{margin:18px 0}.note{background:#fff5dc;padding:15px}</style>',
             '<h1>人员 NPU：预处理、漏检与耗时排查</h1>',
             '<p>状态：'+html.escape(report['status'])+'</p>',
             '<p class="note">四人样例只统计了前景人数，背景仍有人，不能把额外背景框直接算作误报。阈值表用于定位漏检原因；选定参数后仍需独立样例验证。</p>']
    original = report.get('original_findings', {})
    if all(original.get(key) is not None for key in
           ['forward_mean_seconds', 'total_mean_seconds', 'damo_total_mean_seconds']):
        parts.append('<h2>此前真实 NPU 实测记录</h2><p>'
                     f"NPU 前向平均 {original['forward_mean_seconds']:.4f} 秒；"
                     f"NPU 图像处理总耗时平均 {original['total_mean_seconds']:.4f} 秒；"
                     f"DAMO CPU 总耗时平均 {original['damo_total_mean_seconds']:.4f} 秒。"
                     '这是之前的结果，不能当作本次新预处理后的 NPU 性能。</p>')
    parts.extend(['<h2>不需要 NPU 的预处理检查</h2>',
                  '<table><tr><th>图片</th><th>原外部进程预处理 / 秒</th><th>新进程内预处理 / 秒</th></tr>'])
    for row in report.get('preprocessing_checks', {}).get('measurements', []):
        parts.append(f"<tr><td>{html.escape(row['name'])}</td><td>{row['legacy_median_seconds']:.5f}</td><td>{row['in_process_median_seconds']:.5f}</td></tr>")
    parts.append('</table><p>官方输入像素对比（MAE 不是准确率）：'+html.escape(json.dumps(report.get('preprocessing_checks', {}).get('official_input_similarity'), ensure_ascii=False))+'</p>')
    if report.get('error'):
        parts.append('<p class="note">'+html.escape(report['error'])+'</p>')
    parts.append('<h2>真实 NPU 对比与候选框</h2>')
    parts.append('<p>BGR 拉伸是官方样本支持的默认输入；RGB 和 letterbox 只作诊断对照。检测框图片使用置信度 0.5、NMS IoU 0.45。下调置信度或放宽 NMS 后人数增加，仍需看框确认是真人还是重复框。</p>')
    if report.get('hardware_checks'):
        parts.append('<table><tr><th>图片</th><th>输入方式</th><th>人数</th><th>多次稳定</th><th>总耗时 / 秒</th><th>前向 / 秒</th></tr>')
        for row in report['hardware_checks']:
            parts.append(f"<tr><td>{html.escape(row['name'])}</td><td>{html.escape(row['profile'])}</td><td>{row['count']}</td><td>{'是' if row['stable'] else '否'}</td><td>{row['timings']['total_seconds']:.4f}</td><td>{row['timings']['forward_seconds']:.4f}</td></tr>")
        parts.append('</table>')
    for row in report.get('hardware_checks', []):
        parts.append('<details><summary>'+html.escape(f"{row['name']} / {row['profile']} / {row['count']} 人 / {row['timings']['total_seconds']:.4f} 秒")+'</summary>')
        parts.append('<img src="'+row['overlay_file']+'" alt="实际检测框"><table><tr><th>置信度阈值</th><th>NMS IoU 阈值</th><th>人数</th></tr>')
        for setting in row['threshold_matrix']:
            parts.append(f"<tr><td>{setting['threshold']}</td><td>{setting['nms_threshold']}</td><td>{setting['count']}</td></tr>")
        parts.append('</table></details>')
    parts.append('<p>本诊断命令保持当前模型配置。原始张量保存在同目录 npz；各样例 JSON 保存候选框和逐次人数；完整结构化结果见 report.json。</p>')
    (directory/'report.html').write_text(''.join(parts))


def run(args):
    directory = ROOT/'data/npu-diagnostics'/dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    directory.mkdir(parents=True)
    config_before = hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest()
    report = {'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'status': 'running', 'original_findings': original_findings(),
              'reference_limitations': '000000013291 is an incomplete foreground count, not exhaustive scene ground truth.',
              'preprocessing_checks': {},
              'hardware_checks': [], 'configuration_modified': False}
    exit_code = 0
    stage = 'preprocessing'
    try:
        cases = references()
        report['preprocessing_checks'] = preprocessing_checks(cases, args.repeats)
        if args.offline:
            report['status'] = 'preprocessing_checked_npu_not_tested'
        elif not Path('/dev/aipu').exists():
            report.update(status='preprocessing_checked_npu_unavailable',
                          error='当前执行环境没有暴露 /dev/aipu，请在板卡普通终端运行 ./diagnose-person-npu.sh；这不表示板卡驱动缺失。')
            exit_code = 2
        else:
            stage = 'hardware'
            hardware_checks(cases, directory, args.repeats, rows=report['hardware_checks'])
            report['status'] = 'diagnosis_complete_manual_review_required'
    except Exception as exc:
        report.update(status=stage+'_diagnosis_failed', error=str(exc)[-2000:])
        exit_code = 1
    report['active_configuration_unchanged'] = config_before == hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest()
    (directory/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    render_report(report, directory)
    print(json.dumps({'status': report['status'], 'report': str(directory/'report.html'),
                      'configuration_modified': False, 'error': report.get('error')}, ensure_ascii=False), flush=True)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='仅运行图片预处理和旧记录分析，不使用 NPU')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error('repeats 必须为 1..10')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
