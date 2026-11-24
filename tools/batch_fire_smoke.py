#!/usr/bin/env python3
"""Persistent CPU inference through the same run_case used by the web worker."""
import argparse
from collections import Counter
import contextlib
import csv
import datetime as dt
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent
import triage
from tools.detect_fire_smoke import LocalDetector, fingerprint

DATASET = agent.ROOT/'data/datasets/smoke_fire_datasets_1'
EXTRACTED = agent.DATA / 'datasets' / 'smoke_fire_datasets_1'
BATCHES = agent.DATA / 'batches'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def prepare():
    items = []
    archives = []
    for name in ('fire_smoke.zip', 'fire_600.zip'):
        archive = DATASET / name
        archives.append({'path': str(archive), 'sha256': agent.sha256(archive)})
        target = EXTRACTED / archive.stem
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                path = Path(member.filename)
                if member.is_dir() or '__MACOSX' in path.parts or any(p.startswith('.') for p in path.parts):
                    continue
                if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.xml', '.txt'}:
                    continue
                output = (target / path).resolve()
                if not output.is_relative_to(target.resolve()) or member.file_size > 50*1024*1024:
                    raise ValueError('数据集包含无效路径或超大文件')
                if not output.is_file() or output.stat().st_size != member.file_size:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as source, output.open('wb') as destination:
                        import shutil
                        shutil.copyfileobj(source, destination)
        split_by_name = {}
        for split, filename in [('train', 'train_list.txt'), ('val', 'val_list.txt')]:
            split_file = target / 'fire_smoke' / filename
            if split_file.is_file():
                for line in split_file.read_text().splitlines():
                    parts = line.split()
                    if parts:
                        split_by_name[Path(parts[0]).name] = split
        for image in sorted(target.rglob('*')):
            if image.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            annotation = target / 'fire_smoke' / 'annotations' / (image.stem + '.xml')
            items.append({'image': str(image), 'name': image.name, 'archive': name,
                          'annotation': str(annotation) if annotation.is_file() else None,
                          'split': split_by_name.get(image.name, 'unlisted') if annotation.is_file() else 'unlabeled'})
    manifest = {'source': 'https://modelscope.cn/datasets/davidwu123/smoke_fire_datasets_1',
                'dataset_card_note': '默认数据集卡片；来源、标注质量和与预训练数据重叠情况未经核实',
                'archives': archives, 'total': len(items), 'items': items}
    write_json(EXTRACTED / 'manifest.json', manifest)
    return manifest


def read_annotation(path, image_size):
    if Path(path).stat().st_size > 1024*1024:
        raise ValueError('标注文件过大')
    root = ET.parse(path).getroot()
    size = [int(root.findtext('size/width', '0')), int(root.findtext('size/height', '0'))]
    if size != image_size:
        raise ValueError(f'标注尺寸{size}与图片尺寸{image_size}不符')
    objects = []
    for obj in root.findall('object'):
        label = (obj.findtext('name') or '').strip().lower()
        if label not in {'fire', 'smoke'}:
            raise ValueError('未支持的标注类别：' + label)
        values = [float(obj.findtext('bndbox/' + name, 'nan')) for name in ('xmin', 'ymin', 'xmax', 'ymax')]
        # PASCAL VOC's inclusive, one-based coordinates to zero-based xyxy.
        box = [values[0]-1, values[1]-1, values[2], values[3]]
        if not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
            raise ValueError('标注框超出图片或无效')
        objects.append({'label': label, 'box': box, 'difficult': obj.findtext('difficult', '0') == '1'})
    return objects


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0


def evaluate(predictions, objects):
    result = {}
    for label in ('fire', 'smoke'):
        gt = [o for o in objects if o['label'] == label]
        matches = set()
        tp = fp = ignored = 0
        for prediction in sorted((p for p in predictions if p['label'] == label),
                                 key=lambda p: p['confidence'], reverse=True):
            candidates = [(iou(prediction['box'], obj['box']), i) for i, obj in enumerate(gt)]
            overlap, index = max(candidates, default=(0, -1))
            if overlap >= .5 and gt[index]['difficult']:
                ignored += 1
            elif overlap >= .5 and index not in matches:
                matches.add(index)
                tp += 1
            else:
                fp += 1
        result[label] = {'tp': tp, 'fp': fp, 'fn': sum(not o['difficult'] for o in gt)-tp,
                         'ignored_difficult': ignored}
    return result


def empty_metrics():
    return {scope: {label: {'tp': 0, 'fp': 0, 'fn': 0, 'ignored_difficult': 0}
                    for label in ('fire', 'smoke')} for scope in ('all_labeled', 'val')}


def update_metrics(metrics, entry):
    if not entry.get('evaluation'):
        return
    for scope in ['all_labeled'] + (['val'] if entry['split'] == 'val' else []):
        for label, values in entry['evaluation'].items():
            for name, value in values.items():
                metrics[scope][label][name] += value


def metrics_report(metrics):
    result = json.loads(json.dumps(metrics))
    for values in result.values():
        for item in values.values():
            tp, fp, fn = item['tp'], item['fp'], item['fn']
            item.update(precision=tp/(tp+fp) if tp+fp else None,
                        recall=tp/(tp+fn) if tp+fn else None,
                        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    return result


def run(batch_id):
    folder = BATCHES / batch_id
    folder.mkdir(parents=True, exist_ok=True)
    progress_file, results_file = folder/'status.json', folder/'results.jsonl'
    status = {'id': batch_id, 'status': 'preparing', 'started_at': agent.utc_now(),
              'completed': 0, 'total': None, 'errors': 0}
    write_json(progress_file, status)
    manifest = prepare()
    settings = agent.CONFIG['fire_smoke']
    identity = {'detector': fingerprint(settings), 'triage': triage.VERSION,
                'archives': manifest['archives']}
    identity_file = folder/'identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError('模型、评分版本或数据集发生变化，请启动新的批次')
    write_json(identity_file, identity)
    completed = {}
    metrics = empty_metrics()
    levels = Counter()
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            entry = json.loads(line)
            completed[entry['index']] = entry
            update_metrics(metrics, entry)
            levels[str(entry.get('level'))] += 1
    status.update(total=manifest['total'], completed=len(completed), status='running',
                  settings={'threshold': settings['threshold'], 'max_side': settings['max_side'],
                            'iou_threshold': .5, 'triage_version': triage.VERSION},
                  dataset_source=manifest['source'], splits=dict(Counter(i['split'] for i in manifest['items'])),
                  scope='固定阈值下的框级precision/recall/F1；含训练列表，仅val部分单独报告；不是mAP或现场风险准确率')
    write_json(progress_file, status)
    detector = LocalDetector(settings['model_dir'], settings['threshold'], settings['max_side'], settings['threads'])
    started = time.perf_counter()
    with results_file.open('a', encoding='utf-8', buffering=1) as results, (folder/'inference.log').open('a') as log:
        for index, sample in enumerate(manifest['items']):
            if index in completed:
                continue
            entry = dict(sample, index=index)
            try:
                def fire_tool(image):
                    payload = detector.infer(image)
                    return payload['detections'], payload
                args = argparse.Namespace(task='检查图片烟雾明火风险，给出评分和应对建议', image=sample['image'],
                                          observation='', npu_detect=True, detections=None, tool_error=False,
                                          confirm=None, confirm_after=False, llm=False, safety_context={}, batch_id=batch_id)
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    record = agent.run_case(args, fire_detector=fire_tool)
                assessment = record['response_assessment']
                entry.update(case_id=record['case_id'], score=assessment['score'], level=assessment['level'],
                             label=assessment['label'], detections=record['detections'],
                             inference_seconds=(record.get('vision_result') or {}).get('elapsed_seconds'),
                             tool_status=record['vision_tools']['fire_smoke']['status'])
                if entry['tool_status'] != 'ok':
                    entry['error'] = record['tool_error_message']
                elif sample['annotation']:
                    try:
                        objects = read_annotation(sample['annotation'], record['vision_result']['image_size'])
                        entry['evaluation'] = evaluate(record['detections'], objects)
                        entry['ground_truth_count'] = len(objects)
                    except (ValueError, OSError, ET.ParseError) as exc:
                        entry['annotation_error'] = str(exc)
            except Exception as exc:
                entry.update(error=str(exc), score=None, level=None, label='失败，未评分')
            results.write(json.dumps(entry, ensure_ascii=False) + '\n')
            completed[index] = entry
            update_metrics(metrics, entry)
            levels[str(entry.get('level'))] += 1
            status.update(completed=len(completed), errors=sum(bool(e.get('error')) for e in completed.values()),
                          annotation_errors=sum(bool(e.get('annotation_error')) for e in completed.values()),
                          scored=sum(e.get('score') is not None for e in completed.values()),
                          evaluated=sum(bool(e.get('evaluation')) for e in completed.values()),
                          levels=dict(levels), metrics=metrics_report(metrics), last_image=sample['name'],
                          updated_at=agent.utc_now(), elapsed_seconds=round(time.perf_counter()-started, 1))
            if len(completed) % 10 == 0 or len(completed) == manifest['total']:
                write_json(progress_file, status)
                print(f"{len(completed)}/{manifest['total']} 图片，错误{status['errors']}，{status['elapsed_seconds']}秒", flush=True)
    status.update(status='done', finished_at=agent.utc_now())
    write_json(progress_file, status)
    with (folder/'results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['index', 'image', 'split', 'score', 'level', 'label', 'case_id', 'error', 'annotation_error'])
        for entry in completed.values():
            writer.writerow([entry.get(key, '') for key in ('index', 'name', 'split', 'score', 'level', 'label', 'case_id', 'error', 'annotation_error')])
    from tools.finalize_batch import finalize
    finalize(batch_id)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-id', default=dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    import re
    if not re.fullmatch(r'\d{8}-\d{6}', args.batch_id):
        parser.error('批次编号无效')
    if args.prepare_only:
        manifest = prepare()
        print(json.dumps({'total': manifest['total'], 'splits': dict(Counter(i['split'] for i in manifest['items']))}, ensure_ascii=False))
        return
    try:
        run(args.batch_id)
    except (Exception, KeyboardInterrupt) as exc:
        folder = BATCHES/args.batch_id
        status = json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        status.update(id=args.batch_id, status='failed', error=str(exc) or '任务被中断，可用同一批次编号继续', finished_at=agent.utc_now())
        write_json(folder/'status.json', status)
        raise


if __name__ == '__main__':
    main()
