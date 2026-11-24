#!/usr/bin/env python3
"""Install a separate CPU runtime and validate local smoke/fire sample inference."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import venv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent
from tools.detect_fire_smoke import check_model, fingerprint

ROOT = agent.ROOT


def write_marker(settings, value):
    path = Path(settings['validation_file'])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def validate(settings):
    model_dir = Path(settings['model_dir'])
    check_model(model_dir)
    cases = json.loads((ROOT/'samples/fire-smoke/validation_cases.json').read_text())
    results = []
    positive_labels = set()
    for case in cases:
        image = ROOT / case['image']
        cmd = [settings['python'], str(ROOT/'tools/detect_fire_smoke.py'), str(image),
               '--model-dir', str(model_dir), '--threshold', str(settings['threshold']),
               '--max-side', str(settings['max_side']), '--threads', str(settings['threads'])]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout)[-2500:])
        payload = json.loads(result.stdout)
        if payload.get('model_sha256') != check_model(model_dir)['sha256']['damoyolo_tinynasL25_S_smokefire.pt']:
            raise RuntimeError('验证结果的模型版本不匹配')
        if not isinstance(payload.get('detections'), list) or len(payload.get('image_size', [])) != 2:
            raise ValueError('验证结果格式不正确')
        labels = {d['label'] for d in payload['detections']}
        if case['positive']:
            positive_labels.update(labels)
        elif labels:
            raise RuntimeError('普通人物负样例出现烟火误检，请核查权重和类别映射；暂不标记为可用')
        results.append({'image': str(image), 'case': case, 'result': payload})
        print(f"样例 {image.name}：{payload['detections']}，耗时 {payload['elapsed_seconds']} 秒", flush=True)
    if not {'fire', 'smoke'}.issubset(positive_labels):
        raise RuntimeError('本地样例未同时验证到 smoke 和 fire，保留为待验证状态；需核查输出和类别映射')
    marker = {'status': 'passed', 'checked_at': agent.utc_now(), 'fingerprint': fingerprint(settings),
              'samples': results, 'scope': 'sample_inference_only_not_accuracy_evaluation',
              'model_sha256': check_model(model_dir)['sha256']['damoyolo_tinynasL25_S_smokefire.pt']}
    write_marker(settings, marker)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--wheel-dir', type=Path, help='离线安装目录，须包含 ARM64/Python 3.11 所需全部 wheels')
    parser.add_argument('--index-url', default='https://pypi.org/simple')
    args = parser.parse_args()
    settings = agent.CONFIG['fire_smoke']
    try:
        check_model(settings['model_dir'])
        if not args.validate_only:
            runtime = Path(settings['python']).parent.parent
            if not Path(settings['python']).is_file():
                print(f'创建独立烟火环境：{runtime}', flush=True)
                venv.EnvBuilder(with_pip=True).create(runtime)
            command = [settings['python'], '-m', 'pip', 'install', '--retries', '0', '--timeout', '15',
                       '-r', str(ROOT/'tools/fire-smoke-requirements.txt')]
            if args.wheel_dir:
                command += ['--no-index', '--find-links', str(args.wheel_dir.resolve())]
            else:
                command += ['--index-url', args.index_url]
            print('安装烟火 CPU 推理依赖；保留现有 NOE 环境。', flush=True)
            installed = subprocess.run(command)
            if installed.returncode:
                reason = '依赖安装失败，请查看上面的 pip 错误。'
                if args.wheel_dir:
                    reason = '离线依赖安装失败，请检查目录中 wheel 是否齐全且适用于 ARM64/Python 3.11。'
                else:
                    try:
                        from urllib.parse import urlsplit
                        hostname = urlsplit(args.index_url).hostname
                        socket.getaddrinfo(hostname, 443)
                    except OSError:
                        reason = f'依赖安装失败，当前执行环境无法解析 {hostname}，无法下载 PyTorch。'
                raise RuntimeError(reason + '请在能联网的板卡终端重试，或用 --wheel-dir 指定离线依赖目录。')
        validate(settings)
        frozen = subprocess.run([settings['python'], '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
        (ROOT/'data/fire_smoke_runtime_versions.txt').write_text(frozen.stdout)
        print('烟火 CPU 样例推理通过。重启网页服务后可上传图片检查烟雾和明火。')
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        write_marker(settings, {'status': 'failed', 'checked_at': agent.utc_now(),
                                'message': str(exc)[-2500:]})
        print(f'烟火环境尚未就绪：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
