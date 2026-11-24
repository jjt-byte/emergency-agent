#!/usr/bin/env python3
"""Offline CIX YOLOv8s person detection. Native logs go to stderr, JSON to stdout."""
import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __package__:
    from . import npu_image
else:
    import npu_image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT/'models/person/cix-yolov8s/yolov8s.cix'
SOURCE = 'cix_yolov8s_npu'


def fingerprint(settings):
    paths = [Path(settings['model_path']), Path(settings['model_path']).with_name('source.json'),
             Path(__file__), Path(npu_image.__file__), Path(settings['python'])]
    return {'files': {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths},
            'threshold': settings.get('threshold', .5), 'nms_threshold': settings.get('nms_threshold', .45),
            'preprocessing': 'bgr_stretch_640_div255_' + npu_image.VERSION}


def readiness(settings):
    try:
        marker = json.loads(Path(settings['validation_file']).read_text())
        ready = (Path('/dev/aipu').exists() and marker.get('status') == 'passed'
                 and marker.get('fingerprint') == fingerprint(settings))
    except (OSError, ValueError, KeyError):
        ready = False
    return {'ready': ready, 'reason': '本地 CIX YOLOv8s · NPU · 已通过样例对比' if ready
            else 'CIX YOLOv8s · NPU · 待板卡实测验证'}


@contextlib.contextmanager
def native_logs_to_stderr():
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            try:
                yield
            finally:
                ctypes.CDLL(None).fflush(None)
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def prepare_image_ffmpeg(image):
    """Match CIX sample geometry/color/range, using local ffmpeg instead of cv2.

    The bundled input0.bin matches a stretched BGR test image; calibration is 0..1.
    FFmpeg bilinear differs slightly from the original OpenCV resize/JPEG decoder.
    """
    image = Path(image).resolve()
    if not image.is_file():
        raise ValueError('图片不存在：' + str(image))
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=width,height', '-of', 'json', str(image)],
                           capture_output=True, text=True, check=True, timeout=30)
    info = json.loads(probe.stdout)['streams'][0]
    width, height = int(info['width']), int(info['height'])
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-noautorotate', '-i', str(image),
                          '-frames:v', '1', '-vf', 'scale=640:640:flags=bilinear',
                          '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                         capture_output=True, check=True, timeout=30).stdout
    data = np.frombuffer(raw, np.uint8).reshape(640, 640, 3)
    return np.ascontiguousarray(data.transpose(2, 0, 1)[None], dtype=np.float32)/255., (width, height)


def prepare_image(image):
    data, size, _ = npu_image.prepare(image)
    return data, size


def prediction_rows(outputs):
    if len(outputs) != 1:
        raise ValueError('YOLOv8s 必须返回一个输出张量')
    raw = np.asarray(outputs[0])
    if raw.size != 84*8400 or not np.isfinite(raw).all():
        raise ValueError('YOLOv8s 输出尺寸错误或存在无效数值')
    pred = raw.reshape(84, 8400).T
    probabilities = pred[:, 4:]
    if np.any(probabilities < -.001) or np.any(probabilities > 1.001):
        raise ValueError('YOLOv8s 类别置信度超出范围，请检查反量化')
    return pred


def decode(outputs, width, height, threshold=.5, nms_threshold=.45, transform=None):
    pred = prediction_rows(outputs)
    probabilities = pred[:, 4:]
    # Standard best-class YOLO decoding: COCO person is class 0, no objectness field.
    mask = (probabilities.argmax(axis=1) == 0) & (probabilities[:, 0] >= threshold)
    rows = pred[mask]
    if not len(rows):
        return []
    boxes = np.column_stack((rows[:, :2]-rows[:, 2:4]/2, rows[:, :2]+rows[:, 2:4]/2))
    valid = (rows[:, 2] > 0) & (rows[:, 3] > 0)
    boxes, scores = boxes[valid], rows[valid, 4]
    order = np.argsort(-scores, kind='stable')
    keep = []
    while order.size:
        i = int(order[0]); keep.append(i)
        rest = order[1:]
        lt, rb = np.maximum(boxes[i, :2], boxes[rest, :2]), np.minimum(boxes[i, 2:], boxes[rest, 2:])
        intersection = np.maximum(rb-lt, 0).prod(axis=1)
        area = np.maximum(boxes[:, 2:]-boxes[:, :2], 0).prod(axis=1)
        iou = intersection / np.maximum(area[i]+area[rest]-intersection, 1e-9)
        order = rest[iou <= nms_threshold]
    results = []
    for i in keep:
        if transform:
            scale, pad = transform['scale'], transform['pad']
            box = (boxes[i]-np.array([*pad, *pad]))/np.array([*scale, *scale])
        else:
            box = boxes[i]*np.array([width, height, width, height])/640.
        box = np.clip(box, 0, [width, height, width, height])
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        results.append({'label': 'person', 'confidence': round(float(np.clip(scores[i], 0, 1)), 6),
                        'box': [round(float(v), 2) for v in box]})
    return results


class LocalDetector:
    def __init__(self, model_path=DEFAULT_MODEL, threshold=.5, nms_threshold=.45):
        self.model_path = Path(model_path).resolve()
        self.threshold, self.nms_threshold = threshold, nms_threshold
        if not 0 < threshold < 1 or not 0 < nms_threshold < 1:
            raise ValueError('置信度和 NMS 阈值必须介于 0 和 1 之间')
        manifest = json.loads(self.model_path.with_name('source.json').read_text())
        self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if self.model_sha256 != manifest['sha256'].get(self.model_path.name):
            raise ValueError('CIX 模型 SHA256 校验失败')
        if not Path('/dev/aipu').exists():
            raise RuntimeError('当前执行环境看不到 /dev/aipu，请在板卡本机终端运行 NPU 对比脚本')
        with native_logs_to_stderr():
            from NOE_Engine import EngineInfer
            self.engine = EngineInfer(str(self.model_path))

    def infer_raw(self, image, preprocessing='bgr_stretch'):
        """Keep raw NPU output for diagnosis; alternative profiles never activate."""
        preprocess_started = time.perf_counter()
        if preprocessing == 'legacy_ffmpeg':
            data, size = prepare_image_ffmpeg(image)
            transform = {'scale': [640/size[0], 640/size[1]], 'pad': [0, 0]}
        elif preprocessing in {'bgr_stretch', 'rgb_stretch', 'rgb_letterbox'}:
            color, geometry = preprocessing.split('_')
            data, size, transform = npu_image.prepare(image, color.upper(), geometry)
        else:
            raise ValueError('未知的诊断预处理方式')
        preprocess_seconds = time.perf_counter()-preprocess_started
        forward_started = time.perf_counter()
        with native_logs_to_stderr():
            outputs = self.engine.forward([data])
        forward_seconds = time.perf_counter()-forward_started
        # Validate before saving raw data so failed/malformed output never counts as empty.
        prediction_rows(outputs)
        raw = np.asarray(outputs[0]).reshape(1, 84, 8400).copy()
        return raw, {'image_size': list(size), 'preprocess_seconds': round(preprocess_seconds, 6),
                     'forward_seconds': round(forward_seconds, 6), 'transform': transform,
                     'preprocessing': {'color': 'BGR' if preprocessing in {'legacy_ffmpeg', 'bgr_stretch'} else 'RGB',
                                       'range': [0, 1], 'resize': [640, 640],
                                       'geometry': 'letterbox' if preprocessing == 'rgb_letterbox' else 'stretch',
                                       'implementation': 'ffmpeg_bilinear' if preprocessing == 'legacy_ffmpeg'
                                                         else npu_image.VERSION}}

    def infer(self, image):
        started = time.perf_counter()
        raw, details = self.infer_raw(image)
        width, height = details['image_size']
        postprocess_started = time.perf_counter()
        detections = decode([raw], width, height, self.threshold, self.nms_threshold)
        postprocess_seconds = time.perf_counter()-postprocess_started
        return {'detections': detections, 'image_size': [width, height], 'threshold': self.threshold,
                'elapsed_seconds': round(time.perf_counter()-started, 6),
                'forward_seconds': details['forward_seconds'],
                'preprocess_seconds': details['preprocess_seconds'],
                'postprocess_seconds': round(postprocess_seconds, 6), 'source': SOURCE, 'device': 'npu',
                'capabilities': ['person'], 'model_path': str(self.model_path),
                'model_sha256': self.model_sha256,
                'preprocessing': details['preprocessing']}

    def close(self):
        with native_logs_to_stderr():
            self.engine.clean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--nms-threshold', type=float, default=.45)
    args = parser.parse_args()
    started = time.perf_counter()
    detector = LocalDetector(args.model, args.threshold, args.nms_threshold)
    try:
        result = detector.infer(args.image)
    finally:
        detector.close()
    result['total_seconds'] = round(time.perf_counter()-started, 6)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('CIX YOLOv8s 检测失败：' + str(exc), file=sys.stderr)
        sys.exit(1)
