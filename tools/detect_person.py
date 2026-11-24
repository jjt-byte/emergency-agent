#!/usr/bin/env python3
"""Offline DAMO-YOLO human detection; stdout is one JSON object."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time

if __package__:
    from .detect_fire_smoke import digest, offline_only, prepare_pixels, runtime_missing
else:
    from detect_fire_smoke import digest, offline_only, prepare_pixels, runtime_missing

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'models/person/damoyolo-human'
WEIGHT = 'damoyolo_tinynasL25_S_person.pt'
MODEL_FILES = (WEIGHT, 'damoyolo.py', 'configuration.json', 'label_map.pkl', 'tinynas_L25_k1kx.txt')
SOURCE = 'damoyolo_person_cpu'


def check_model(model_dir):
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / 'source.json').read_text())
    for name in MODEL_FILES:
        path = model_dir / name
        if not path.is_file() or digest(path) != manifest['sha256'].get(name):
            raise ValueError(f'人体模型文件缺失或校验失败：{name}')
    return manifest


def fingerprint(settings):
    paths = [Path(settings['model_dir']) / name for name in (*MODEL_FILES, 'source.json')]
    paths += [Path(__file__), Path(__file__).with_name('detect_fire_smoke.py'), Path(settings['python'])]
    return {'files': {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths},
            'threshold': settings.get('threshold', .5), 'max_side': settings.get('max_side', 640)}


def readiness(settings):
    try:
        marker = json.loads(Path(settings['validation_file']).read_text())
        ready = marker.get('status') == 'passed' and marker.get('fingerprint') == fingerprint(settings)
    except (OSError, ValueError, KeyError):
        ready = False
    return {'ready': ready, 'reason': '本地 DAMO-YOLO 人体检测 · CPU · 已通过样例推理' if ready
            else '人体检测模型或推理环境待验证'}


def normalize_result(output, width, height, scales=(1.0, 1.0), threshold=.5):
    labels, scores, boxes = output['class_ids'], output['scores'], output['boxes']
    if labels is None and scores is None and boxes is None:
        return []
    if labels is None or scores is None or boxes is None or not len(labels) == len(scores) == len(boxes):
        raise ValueError('人体检测返回的类别、分数和坐标长度不一致')
    results = []
    for label, score, box in zip(labels, scores, boxes):
        if str(label).lower() != 'person':
            raise ValueError(f'人体检测模型出现未知类别：{label}')
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('人体检测返回无效置信度')
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError('人体检测返回无效坐标')
        if score < threshold:
            continue
        x1, y1, x2, y2 = [float(v) / scales[i % 2] for i, v in enumerate(box)]
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        results.append({'label': 'person', 'confidence': round(score, 6),
                        'box': [round(v, 2) for v in (x1, y1, x2, y2)]})
    return sorted(results, key=lambda item: item['confidence'], reverse=True)


class LocalDetector:
    def __init__(self, model_dir=DEFAULT_MODEL, threshold=.5, max_side=640, threads=8):
        self.model_dir = Path(model_dir).resolve()
        self.manifest = check_model(self.model_dir)
        missing = runtime_missing()
        if missing:
            raise RuntimeError('人体推理环境缺少：' + ', '.join(missing))
        os.environ.update(HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1', TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
        with offline_only(), contextlib.redirect_stdout(sys.stderr):
            import torch
            from modelscope.models.cv.tinynas_detection.tinynas_damoyolo import DamoYolo
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(1)
            self.model = DamoYolo(str(self.model_dir), device='cpu', trust_remote_code=True).eval()
            if self.model.num_classes != 1 or self.model.label_map != {1: [{'name': 'person'}]}:
                raise ValueError('模型必须仅包含人体 person 类别')
            self.model.conf_thre = threshold
        self.threshold, self.max_side = threshold, max_side

    def infer(self, image):
        started = time.perf_counter()
        with offline_only(), contextlib.redirect_stdout(sys.stderr):
            import torch
            from PIL import Image
            with Image.open(image) as source:
                source.load()
                rgb = source.convert('RGB')
            width, height = rgb.size
            pixels, scales = prepare_pixels(rgb, self.max_side)
            with torch.inference_mode():
                output = self.model(torch.from_numpy(pixels))
            detections = normalize_result(output, width, height, scales, self.threshold)
        return {'detections': detections, 'image_size': [width, height],
                'threshold': self.threshold, 'elapsed_seconds': round(time.perf_counter()-started, 3),
                'source': SOURCE, 'device': 'cpu', 'capabilities': ['person'],
                'model_path': str(self.model_dir / WEIGHT), 'model_sha256': self.manifest['sha256'][WEIGHT],
                'preprocessing': {'rgb_range': [0, 255], 'stride': 32, 'max_side': self.max_side,
                                  'scales': list(scales)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL))
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--max-side', type=int, default=640)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    try:
        if not 0 < args.threshold < 1 or not 32 <= args.max_side <= 1920 or not 1 <= args.threads <= 32:
            raise ValueError('阈值、图像尺寸或线程数无效')
        started = time.perf_counter()
        detector = LocalDetector(args.model_dir, args.threshold, args.max_side, args.threads)
        result = detector.infer(args.image)
        result['total_seconds'] = round(time.perf_counter()-started, 3)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f'人体检测失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
