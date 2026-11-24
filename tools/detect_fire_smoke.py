#!/usr/bin/env python3
"""Local DAMO-YOLO smoke/fire inference. Stdout is a single JSON object."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'models/fire-smoke/damoyolo-smokefire'
WEIGHT = 'damoyolo_tinynasL25_S_smokefire.pt'
MODEL_FILES = (WEIGHT, 'damoyolo.py', 'label_map.pkl', 'tinynas_L25_k1kx.txt')
RUNTIME_MODULES = ('torch', 'torchvision', 'modelscope', 'numpy', 'PIL', 'easydict',
                   'addict', 'timm', 'thop', 'networkx')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def check_model(model_dir):
    """Only load the reviewed copy of the downloaded model/configuration."""
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / 'source.json').read_text())
    for name in MODEL_FILES:
        path = model_dir / name
        if not path.is_file() or digest(path) != manifest['sha256'].get(name):
            raise ValueError(f'模型文件缺失或校验失败：{name}')
    return manifest


def runtime_missing():
    return [name for name in RUNTIME_MODULES if importlib.util.find_spec(name) is None]


def fingerprint(settings):
    """Cheap identity for a validation marker; inference checks full hashes too."""
    paths = [Path(settings['model_dir']) / name for name in MODEL_FILES]
    paths += [Path(settings['model_dir']) / 'source.json', Path(__file__), Path(settings['python'])]
    return {'files': {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths},
            'threshold': settings.get('threshold', .6), 'max_side': settings.get('max_side', 640)}


def readiness(settings):
    if not settings or not settings.get('enabled'):
        return {'ready': False, 'weights': False, 'reason': '尚未启用烟火检测'}
    weights = (Path(settings['model_dir']) / WEIGHT).is_file()
    try:
        marker = json.loads(Path(settings['validation_file']).read_text())
        ready = marker.get('status') == 'passed' and marker.get('fingerprint') == fingerprint(settings)
    except (OSError, ValueError, KeyError):
        marker, ready = {}, False
    reason = ('本地 DAMO-YOLO · CPU · 已通过样例推理' if ready else
              marker.get('message', '权重已准备，推理环境待安装和验证') if weights else '尚未下载完整权重')
    return {'ready': ready, 'weights': weights, 'reason': reason}


def prepare_pixels(rgb, max_side=640):
    """Match upstream RGB 0..255 CHW and stride-32 padding; bound image size."""
    import numpy as np
    from PIL import Image
    width, height = rgb.size
    scale = min(1.0, max_side / max(width, height)) if max_side else 1.0
    resized_width, resized_height = max(1, int(width * scale)), max(1, int(height * scale))
    if (resized_width, resized_height) != rgb.size:
        rgb = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    image = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)
    padded = np.zeros((1, 3, math.ceil(resized_height / 32) * 32,
                       math.ceil(resized_width / 32) * 32), dtype=np.float32)
    padded[0, :, :resized_height, :resized_width] = image
    return padded, (resized_width / width, resized_height / height)


def normalize_result(output, width, height, scales=(1.0, 1.0), threshold=.6):
    """The upstream detector returns label names, not the model card's IDs."""
    labels, scores, boxes = output['class_ids'], output['scores'], output['boxes']
    if labels is None and scores is None and boxes is None:
        return []
    if not (len(labels) == len(scores) == len(boxes)):
        raise ValueError('烟火检测返回的类别、分数和坐标长度不一致')
    results = []
    for label, score, box in zip(labels, scores, boxes):
        label, score = str(label).lower(), float(score)
        if label == 'background':
            continue
        if label not in {'fire', 'smoke'}:
            raise ValueError(f'烟火模型出现未知类别：{label}')
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('烟火检测返回无效置信度')
        if score < threshold:
            continue
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError('烟火检测返回无效坐标')
        x1, y1, x2, y2 = [float(v) / scales[i % 2] for i, v in enumerate(box)]
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        results.append({'label': label, 'confidence': round(score, 6),
                        'box': [round(v, 2) for v in (x1, y1, x2, y2)]})
    return sorted(results, key=lambda item: item['confidence'], reverse=True)[:100]


@contextlib.contextmanager
def offline_only():
    """Reject connections if a framework tries to fetch anything at runtime."""
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    def blocked(*args, **kwargs):
        raise RuntimeError('烟火推理只允许本地资源，已阻止网络连接')
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex = original_connect, original_connect_ex


class LocalDetector:
    """Load weights once for a dataset; use identical per-image inference."""

    def __init__(self, model_dir=DEFAULT_MODEL, threshold=.6, max_side=640, threads=8):
        self.model_dir = Path(model_dir).resolve()
        self.manifest = check_model(self.model_dir)
        missing = runtime_missing()
        if missing:
            raise RuntimeError('烟火推理环境缺少：' + ', '.join(missing))
        os.environ.update(HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
                          TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
        with offline_only(), contextlib.redirect_stdout(sys.stderr):
            import torch
            from modelscope.models.cv.tinynas_detection.tinynas_damoyolo import DamoYolo
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(1)
            self.model = DamoYolo(str(self.model_dir), device='cpu', trust_remote_code=True).eval()
            self.model.conf_thre = threshold
        self.threshold, self.max_side = threshold, max_side

    def infer(self, image):
        return infer(image, self.model_dir, self.threshold, self.max_side, _loaded=self)


def infer(image, model_dir=DEFAULT_MODEL, threshold=.6, max_side=640, threads=8, _loaded=None):
    image, model_dir = Path(image).resolve(), Path(model_dir).resolve()
    if not image.is_file():
        raise ValueError('本地图片不存在')
    manifest = _loaded.manifest if _loaded else check_model(model_dir)
    missing = runtime_missing()
    if missing:
        raise RuntimeError('烟火推理环境缺少：' + ', '.join(missing) +
                           '。请在板卡终端运行 ./setup-fire-smoke.sh')
    os.environ.update(HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
                      TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
    started = time.perf_counter()
    # Bypass the general Pipeline/datasets stack. DamoYolo reads the reviewed
    # local damoyolo.py (correct weight), not the stale configuration.json name.
    with offline_only(), contextlib.redirect_stdout(sys.stderr):
        import torch
        from PIL import Image
        from modelscope.models.cv.tinynas_detection.tinynas_damoyolo import DamoYolo
        if not _loaded:
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(1)
        with Image.open(image) as source:
            source.load()
            rgb = source.convert('RGB')
        width, height = rgb.size
        pixels, scales = prepare_pixels(rgb, max_side)
        model = _loaded.model if _loaded else DamoYolo(str(model_dir), device='cpu', trust_remote_code=True).eval()
        model.conf_thre = threshold
        with torch.inference_mode():
            output = model(torch.from_numpy(pixels))
        detections = normalize_result(output, width, height, scales, threshold)
    return {'detections': detections, 'image_size': [width, height],
            'threshold': threshold, 'elapsed_seconds': round(time.perf_counter() - started, 3),
            'source': 'damoyolo_cpu', 'device': 'cpu', 'capabilities': ['fire', 'smoke'],
            'model_sha256': manifest['sha256'][WEIGHT],
            'preprocessing': {'rgb_range': [0, 255], 'stride': 32, 'max_side': max_side,
                              'scales': list(scales)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?')
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL))
    parser.add_argument('--threshold', type=float, default=.6)
    parser.add_argument('--max-side', type=int, default=640)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        if not 0 < args.threshold < 1 or not 32 <= args.max_side <= 1920 or not 1 <= args.threads <= 32:
            raise ValueError('阈值、图像尺寸或线程数无效')
        if args.check:
            check_model(args.model_dir)
            missing = runtime_missing()
            print(json.dumps({'weights_valid': True, 'runtime_ready': not missing,
                              'missing_modules': missing}, ensure_ascii=False))
            return 1 if missing else 0
        if not args.image:
            parser.error('请提供本地图片路径')
        result = infer(args.image, args.model_dir, args.threshold, args.max_side, args.threads)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f'烟火检测失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
