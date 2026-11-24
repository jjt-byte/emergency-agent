#!/usr/bin/env python3
"""Run real offline CPU smoke/person checks and generate fresh local readiness."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agent
from tools.detect_person import LocalDetector, fingerprint
from tools.setup_fire_smoke import validate, write_marker


def main():
    config = agent.CONFIG
    settings = config['person_detection']
    if Path(settings['python']).resolve() != Path(sys.executable).resolve():
        raise RuntimeError('请用 config.json 中的推理 Python 运行验证')
    started = time.perf_counter()
    try:
        detector = LocalDetector(settings['model_dir'], settings['threshold'], settings['max_side'], settings['threads'])
        results = []
        for name, image, count in [
            ('previous_three_person_image', ROOT/'samples/000000252219.jpg', 3),
            ('blank_negative', ROOT/'samples/validation-blank.png', 0),
        ]:
            result = detector.infer(image)
            results.append(dict(result, name=name, image=str(image), expected_count=count))
            if len(result['detections']) != count:
                raise RuntimeError(f'{name}: 检出 {len(result["detections"])}，参考 {count}；暂不启用')
        write_marker(settings, {'status': 'passed', 'fingerprint': fingerprint(settings),
                     'checked_at': agent.utc_now(), 'results': results,
                     'scope': 'two sample inference checks, not dataset accuracy'})
    except Exception as exc:
        write_marker(settings, {'status': 'failed', 'message': str(exc)})
        raise
    finally:
        if 'detector' in locals():
            del detector
    try:
        validate(config['fire_smoke'])
    except Exception as exc:
        write_marker(config['fire_smoke'], {'status': 'failed', 'message': str(exc)})
        raise
    completion = subprocess.run([config['llama_cli'], '-m', config['model_path'], '-t',
        str(config['threads']), '-c', '512', '-n', '8', '--no-display-prompt',
        '-p', 'Reply briefly in Chinese: hello.'], capture_output=True, text=True, timeout=180)
    if completion.returncode or not completion.stdout.strip():
        raise RuntimeError('LLM 冒烟测试失败: ' + completion.stderr[-1200:])
    (ROOT/'data/deployment_validation.json').write_text(json.dumps({
        'checked_at': agent.utc_now(), 'status': 'passed', 'device': 'CPU',
        'elapsed_seconds': round(time.perf_counter()-started, 3),
        'llm_output': completion.stdout[-500:], 'scope': 'deployment smoke test only'
    }, ensure_ascii=False, indent=2)+'\n')
    print('CPU 人员、烟火和本地 LLM 冒烟验证通过；不代表完整精度评估。')


if __name__ == '__main__':
    main()
