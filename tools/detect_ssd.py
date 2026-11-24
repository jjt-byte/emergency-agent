#!/usr/bin/env python3
"""Run the bundled CIX SSD COCO detector on one image and print JSON."""
import argparse
import contextlib
import json
from math import sqrt
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "ssd_300.cix"

# Adapted from NVIDIA's Apache-2.0 SSD decoder (2018-2019).
# See models/SSD-NOTICE.txt for attribution and source information.


def default_boxes():
    feature_sizes = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    boxes = []
    for layer, feat in enumerate(feature_sizes):
        fk = 300 / steps[layer]
        scale = scales[layer] / 300
        next_scale = scales[layer + 1] / 300
        sizes = [(scale, scale), (sqrt(scale * next_scale),) * 2]
        for ratio in aspect_ratios[layer]:
            width, height = scale * sqrt(ratio), scale / sqrt(ratio)
            sizes.extend([(width, height), (height, width)])
        for width, height in sizes:
            for height_index in range(feat):
                for width_index in range(feat):
                    center_x = (width_index + 0.5) / fk
                    center_y = (height_index + 0.5) / fk
                    boxes.append((center_x, center_y, width, height))
    boxes = np.asarray(boxes, dtype=np.float32)
    boxes = np.clip(boxes, 0, 1)
    centers = boxes[:, :2]
    sizes = boxes[:, 2:]
    return np.column_stack((centers - sizes / 2, centers + sizes / 2))


ANCHORS = default_boxes()


def decode_boxes(localizations):
    centers = (localizations[:, :2] * 0.1) * (ANCHORS[:, 2:] - ANCHORS[:, :2])
    centers += (ANCHORS[:, :2] + ANCHORS[:, 2:]) / 2
    sizes = np.exp(localizations[:, 2:] * 0.2) * (ANCHORS[:, 2:] - ANCHORS[:, :2])
    return np.column_stack((centers - sizes / 2, centers + sizes / 2))


def nms(boxes, scores, threshold=0.5, limit=20):
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size and len(keep) < limit:
        current = int(order[0])
        keep.append(current)
        rest = order[1:]
        if not rest.size:
            break
        left_top = np.maximum(boxes[current, :2], boxes[rest, :2])
        right_bottom = np.minimum(boxes[current, 2:], boxes[rest, 2:])
        intersection_size = np.maximum(right_bottom - left_top, 0)
        intersection = intersection_size[:, 0] * intersection_size[:, 1]
        area_current = np.prod(np.maximum(boxes[current, 2:] - boxes[current, :2], 0))
        area_rest = np.prod(np.maximum(boxes[rest, 2:] - boxes[rest, :2], 0), axis=1)
        iou = intersection / np.maximum(area_current + area_rest - intersection, 1e-9)
        order = rest[iou < threshold]
    return keep


def decode(localizations, logits, width, height, threshold):
    boxes = decode_boxes(localizations)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    results = []
    # COCO's class index 1 is person; index 0 is background.
    names = ("background person bicycle car motorcycle airplane bus train truck boat "
             "traffic_light fire_hydrant stop_sign parking_meter bench bird cat dog horse "
             "sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase "
             "frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
             "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple "
             "sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant "
             "bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven "
             "toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush").split()
    for class_id in range(1, probabilities.shape[1]):
        scores = probabilities[:, class_id]
        candidates = np.flatnonzero(scores > threshold)
        if not candidates.size:
            continue
        candidates = candidates[np.argsort(scores[candidates])[-200:]]
        for idx in nms(boxes[candidates], scores[candidates]):
            anchor_idx = int(candidates[idx])
            box = np.clip(boxes[anchor_idx], 0, 1)
            results.append({
                "label": names[class_id],
                "confidence": round(float(scores[anchor_idx]), 6),
                "box": [round(float(box[0] * width), 2), round(float(box[1] * height), 2),
                        round(float(box[2] * width), 2), round(float(box[3] * height), 2)],
            })
    results.sort(key=lambda result: result["confidence"], reverse=True)
    return results[:50]


def load_rgb(path):
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
               "-vf", "scale=300:300", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(command, capture_output=True, check=True, timeout=30).stdout
    pixels = np.frombuffer(raw, dtype=np.uint8)
    if pixels.size != 300 * 300 * 3:
        raise ValueError(f"图片读取失败: {path}")
    data = pixels.reshape(300, 300, 3).astype(np.float32)
    return ((data - 128.0) / 128.0).transpose(2, 0, 1)[None, ...].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("threshold 必须在 0 与 1 之间")
    image = Path(args.image).expanduser().resolve()
    if not image.is_file():
        parser.error(f"图片不存在: {image}")
    size = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height", "-of", "csv=s=x:p=0", str(image)],
                          capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    width, height = map(int, size.split("x"))

    inputs = load_rgb(image)
    start = time.perf_counter()
    # NOE logs through both Python and native stdout. Keep stdout valid JSON.
    sys.stdout.flush()
    saved_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            from NOE_Engine import EngineInfer
            engine = EngineInfer(args.model)
            try:
                outputs = engine.forward(inputs)
            finally:
                engine.clean()
            import ctypes
            ctypes.CDLL(None).fflush(None)
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
    localizations = np.asarray(outputs[0]).reshape(1, 4, 8732)[0].T
    logits = np.asarray(outputs[1]).reshape(1, 81, 8732)[0].T
    print(json.dumps({"detections": decode(localizations, logits, width, height, args.threshold),
                      "image_size": [width, height], "threshold": args.threshold,
                      "elapsed_seconds": round(time.perf_counter() - start, 3)}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as exc:
        print(f"检测失败: {exc}", file=sys.stderr)
        sys.exit(2)
