#!/usr/bin/env python3
"""Prepare fixed-shape TorchScript and calibration inputs for CixBuilder on x86.

This exports a candidate graph, not a compiled or NPU-validated .cix model.
"""
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fixed_pixels(image, size=640):
    """Keep current DAMO RGB 0..255 resize, then pad to one fixed square.

    Padding is at the bottom/right; never use the official YOLO BGR/div255
    preprocessing for these DAMO graphs.
    """
    import numpy as np
    from PIL import Image
    from tools.detect_fire_smoke import prepare_pixels
    with Image.open(image) as opened:
        rgb = opened.convert('RGB')
        pixels, scales = prepare_pixels(rgb, size)
        result = np.zeros((1, 3, size, size), dtype=np.float32)
        result[:, :, :pixels.shape[2], :pixels.shape[3]] = pixels
        return result, scales


def image_list(path):
    """Read the local VOC train-list format: image path [annotation path]."""
    path = Path(path).resolve()
    values = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        image = Path(line.split()[0])
        image = image if image.is_absolute() else path.parent/image
        if not image.is_file():
            raise ValueError('校准图片不存在：' + str(image))
        if image.resolve() not in values:
            values.append(image.resolve())
    if not values:
        raise ValueError('校准列表没有图片')
    return values


def build_cfg(name, input_name, output_name, size):
    # model_type=torch and graph input/output names follow the SDK's bundled
    # torch_centernet_resnet50 example. There are no copied node-number overrides.
    return f'''[Common]
mode = build

[Parser]
model_type = torch
model_name = {name}
model_domain = OBJECT_DETECTION
detection_postprocess =
input_model = model/{name}.pt
input = {input_name}
output = {output_name}
input_shape = [1, 3, {size}, {size}]
input_dtype = float32
input_data_format = NCHW
output_dir = ./out

[Optimizer]
dataset = NumpyDataset
calibration_data = datasets/calibration_data.npy
calibration_batch_size = 1
quantize_method_for_activation = per_tensor_asymmetric
quantize_method_for_weight = per_channel_symmetric_restricted_range
activation_bits = 8
weight_bits = 8
bias_bits = 32
output_dir = ./out
dump_dir = ./out

[GBuilder]
target = X2_1204MP3
outputs = {name}.cix
tiling = fps
'''


def export(args):
    import numpy as np
    import torch
    from tools import detect_person, detect_fire_smoke

    settings = json.loads((ROOT/'config.json').read_text())
    config = settings['person_detection'] if args.model == 'person' else settings['fire_smoke']
    if args.model == 'person' and config.get('backend') != 'damoyolo':
        config = settings.get('person_detection_cpu_fallback', {})
    model_dir = Path(config['model_dir'])
    adapter = detect_person if args.model == 'person' else detect_fire_smoke
    adapter.check_model(model_dir)
    images = image_list(args.calibration_list)
    # Deterministically spread calibration examples across the whole list.
    indices = np.linspace(0, len(images)-1, min(args.calibration_limit, len(images)), dtype=int)
    calibration = [images[index] for index in indices]
    if args.validation_list:
        validation = image_list(args.validation_list)
    else:
        marker = json.loads(Path(config['validation_file']).read_text())
        validation = [Path(row['image']) for row in marker.get('results', marker.get('samples', []))]
    if not validation:
        raise ValueError('需要至少一张独立的导出一致性检查图片')
    validation_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in validation}
    if any(hashlib.sha256(p.read_bytes()).hexdigest() in validation_hashes for p in calibration):
        raise ValueError('校准图片不能同时作为本次导出的一致性验证图片')

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        raise ValueError('输出目录已有导出记录，请选择新目录以保留原验证结果')
    (output/'model').mkdir(exist_ok=True)
    (output/'datasets').mkdir(exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1', TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    from modelscope.models.cv.tinynas_detection.tinynas_damoyolo import DamoYolo

    class Graph(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.backbone = detector.backbone
            self.neck = detector.neck
            self.head = detector.head

        def forward(self, images):
            scores, boxes = self.head(self.neck(self.backbone(images)))
            return torch.cat((boxes, scores), dim=-1)

    name = 'damo_person' if args.model == 'person' else 'damo_fire_smoke'
    model_path = output/'model'/f'{name}.pt'
    with detect_fire_smoke.offline_only(), contextlib.redirect_stdout(sys.stderr):
        detector = DamoYolo(str(model_dir), device='cpu', trust_remote_code=True).eval()
        graph = Graph(detector).eval()
        example = torch.from_numpy(fixed_pixels(validation[0], args.size)[0])
        with torch.inference_mode():
            graph(example)  # Set the fixed feature-grid cache before tracing.
            traced = torch.jit.trace(graph, example, check_trace=True)
            traced = torch.jit.freeze(traced)
            traced.save(str(model_path))
            restored = torch.jit.load(str(model_path)).eval()
            parity = []
            for image in validation:
                pixels, _ = fixed_pixels(image, args.size)
                tensor = torch.from_numpy(pixels)
                eager = graph(tensor)
                actual = restored(tensor)
                torch.testing.assert_close(actual, eager, rtol=1e-4, atol=1e-3)
                parity.append({'image_name': image.name,
                               'output_shape': list(actual.shape),
                               'max_absolute_error': float((eager-actual).abs().max())})
                print(f'导出一致性：{image.name} / 最大绝对误差 {parity[-1]["max_absolute_error"]:.6g}', file=sys.stderr)

    inputs = list(restored.graph.inputs())
    input_name = inputs[1].debugName()
    output_name = list(restored.graph.outputs())[0].debugName()
    (output/'build.cfg').write_text(build_cfg(name, input_name, output_name, args.size))
    array = np.lib.format.open_memmap(output/'datasets/calibration_data.npy', mode='w+',
                                      dtype=np.float32, shape=(len(calibration), 3, args.size, args.size))
    for index, image in enumerate(calibration):
        array[index] = fixed_pixels(image, args.size)[0][0]
    array.flush()
    del array
    provenance = [{'image_name': image.name, 'sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
                  for image in calibration]
    manifest = {
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'torchscript_parity_passed_cix_compilation_pending',
        'model': name, 'source_weight_sha256': adapter.check_model(model_dir)['sha256'][adapter.WEIGHT],
        'torch_version': str(torch.__version__), 'graph_file': f'model/{name}.pt',
        'graph_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'input': {'name': input_name, 'shape': [1, 3, args.size, args.size], 'dtype': 'float32',
                  'color': 'RGB', 'range': [0, 255], 'resize': 'downscale longest side, no upscale',
                  'padding': 'bottom/right zero padding to fixed square'},
        'output': {'name': output_name, 'shape': parity[0]['output_shape'],
                   'layout': 'batch, candidates, [x1,y1,x2,y2,class_scores...]',
                   'classes': [detector.label_map[index+1][0]['name'] for index in range(detector.num_classes)],
                   'postprocess': 'CPU: per-class threshold and NMS; ignore background; undo resize; clip to original image'},
        'checks': parity, 'calibration_samples': provenance,
        'npu_target': 'X2_1204MP3', 'cix_compiled': False, 'npu_validated': False,
        'limits': ['Only fixed-square eager vs TorchScript outputs were compared.',
                   'Fixed square padding differs from the current variable-shape CPU path; detection accuracy needs separate testing.',
                   'Calibration is a starter subset, not a representative production calibration or independent accuracy benchmark.',
                   'Compiler operator/quantization compatibility and NPU results have not been tested.']}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    (output/'README.txt').write_text(
        '这是可编译候选包，不是已生成的 CIX/NPU 模型。\n'
        '在 Linux x86_64 + Python 3.10 的 CixBuilder 6.1.3753.3 环境中，进入本目录执行：\n'
        '  cixbuild build.cfg\n'
        '请保留完整编译日志；若算子不支持，须按日志适配或转 ONNX，不能把失败视为已部署。\n'
        '编译产物只有通过板端 NOE 推理和量化前后对比，才能接入网页。\n'
        '当前适配器 tools/detect_yolov8_npu.py 只支持官方 COCO YOLOv8，不能直接加载本 DAMO 模型。\n'
        '预处理和类别顺序见 manifest.json；烟火类别包含 background，必须排除。\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['person', 'fire_smoke'])
    parser.add_argument('--calibration-list', type=Path, required=True)
    parser.add_argument('--validation-list', type=Path)
    parser.add_argument('--calibration-limit', type=int, default=32)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.calibration_limit <= 512 or args.size != 640:
        parser.error('校准样本数须为 1..512；当前仅验证 640×640 固定输入')
    result = export(args)
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()),
                      'validation_images': len(result['checks']),
                      'calibration_images': len(result['calibration_samples'])}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print('CIX 导出准备失败：' + str(exc), file=sys.stderr)
        sys.exit(1)
