#!/usr/bin/env python3
"""Offline web UI with optional LAN access using Python's standard library."""
import argparse
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import mimetypes
import os
from pathlib import Path
import platform
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

import agent
import monitor
import triage
from tools.detect_fire_smoke import readiness as fire_smoke_readiness
from tools.detect_person import readiness as person_readiness
from tools.detect_yolov8_npu import readiness as yolo_readiness

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / 'data' / 'web_jobs'
UPLOADS = ROOT / 'data' / 'uploads'
WEB = ROOT / 'web'
MAX_BODY = 16 * 1024 * 1024
BATCHES = ROOT / 'data' / 'batches'
CONVERSATIONS = ROOT / 'data' / 'conversations'
LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='offline-agent')
RUNNING_JOBS = {}
JOB_TIMEOUT = 900
MONITOR = monitor.Controller(ROOT)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def conversation_path(conversation_id):
    if not isinstance(conversation_id, str) or not re.fullmatch(r'[a-f0-9]{32}', conversation_id):
        raise ValueError('对话编号无效，请开始新对话')
    return CONVERSATIONS / (conversation_id + '.json')


def conversation_context(session, image=None, clear_scene=False):
    """Read trusted local reports; clients cannot supply memories or image paths."""
    records = [read_json(record_path(case_id)) for case_id in session.get('case_ids', [])[-12:]
               if record_path(case_id).is_file()]
    # Legacy reports need an answer for memory, but existing answers must survive.
    records = [record if isinstance(record.get('task_answer'), str) and record['task_answer']
               else agent.with_task_result(record) for record in records]
    context = {'conversation_id': session['id'],
               'history': [{'question': r['task'], 'answer': r['task_answer'][:800]}
                           for r in records[-4:]], 'cached_tools': {}}
    scene = None
    if not image and not clear_scene and session.get('scene_case_id'):
        scene = read_json(record_path(session['scene_case_id']))
        previous = Path(scene['image']['path']).resolve()
        if not previous.is_relative_to(agent.EVIDENCE.resolve()) or not previous.is_file():
            raise ValueError('上一张图片已不可用，请重新上传')
        image = previous
        context.update(scene_reused=True, previous_observation=scene.get('observation', ''),
                       scene_source_case_id=scene['case_id'])
    if image:
        image_hash = agent.sha256(image)
        context['image_sha256'] = image_hash
        for record in records:
            if (record.get('image') or {}).get('sha256') != image_hash:
                continue
            for name, state in record.get('vision_tools', {}).items():
                if state.get('status') == 'ok' and name in record.get('agent_tool_fingerprints', {}):
                    context['cached_tools'][name] = {'state': state, 'case_id': record['case_id'],
                        'fingerprint': record['agent_tool_fingerprints'][name]}
    return context, image


def conversation_view(conversation_id):
    session = read_json(conversation_path(conversation_id))
    latest = read_json(record_path(session['case_ids'][-1])) if session.get('case_ids') else None
    return {'id': session['id'], 'latest_record': agent.with_task_result(latest) if latest else None}


def capabilities():
    # Workers read configuration afresh; keep the long-running status view aligned.
    config = read_json(ROOT/'config.json')
    db = agent.connect()
    count = db.execute('SELECT count(*) FROM docs').fetchone()[0]
    db.close()
    fire_dir = ROOT / 'models' / 'fire-smoke'
    fire_status = fire_smoke_readiness(config.get('fire_smoke', {}))
    person = config.get('person_detection', {})
    if person.get('backend') == 'damoyolo':
        person_status = person_readiness(person)
    elif person.get('backend') == 'cix_yolov8s':
        person_status = yolo_readiness(person)
    else:
        person_status = {'ready': Path('/dev/aipu').exists() and Path(config['vision_model']).is_file(),
                         'reason': '本地 SSD 模型 · NPU' if Path('/dev/aipu').exists() else '当前环境未发现 NPU 设备'}
    return {'npu_device': Path('/dev/aipu').exists(),
            'person_model': agent.person_model_path(person).is_file(),
            'person_ready': person_status['ready'], 'person_status': person_status['reason'],
            'llm_model': Path(config['model_path']).is_file(),
            'llm_runtime': Path(config['llama_cli']).is_file(),
            'agent_enabled': config.get('agent', {}).get('enabled', False),
            'agent_version': 'local-tool-agent-v1',
            'knowledge_count': count, 'fire_smoke_ready': fire_status['ready'],
            'fire_smoke_weights': fire_status['weights'], 'fire_smoke_status': fire_status['reason'],
            'fire_smoke_files': sorted(str(p.relative_to(fire_dir)) for p in fire_dir.rglob('*') if p.suffix in {'.pt', '.onnx', '.cix'}),
            'capabilities': (['person'] if person_status['ready'] else []) + (['fire', 'smoke'] if fire_status['ready'] else []),
            'missing_capabilities': ([] if person_status['ready'] else ['person'])
                + ([] if fire_status['ready'] else ['fire', 'smoke'])}


def evaluation_snapshot(config):
    """Expose only measurements already present in local validation artifacts."""
    def load(path):
        try:
            return read_json(path)
        except (OSError, ValueError, TypeError):
            return None

    def mapping(value):
        return value if isinstance(value, dict) else {}

    def rows(value):
        return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []

    def number(value):
        return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None

    cpu_name = platform.processor() or '本机 CPU'
    try:
        for line in Path('/proc/cpuinfo').read_text(errors='replace').splitlines():
            if line.lower().startswith(('model name', 'hardware', 'model')) and ':' in line:
                candidate = line.split(':', 1)[1].strip()
                if candidate:
                    cpu_name = candidate
                    break
    except OSError:
        pass

    person = config.get('person_detection', {})
    person_path = Path(person.get('validation_file') or ROOT/'data/person_detection_validation.json')
    person_validation = mapping(load(person_path))
    person_results = rows(person_validation.get('results'))
    person_times = [number(row.get('elapsed_seconds')) for row in person_results]
    person_times = [value for value in person_times if value is not None]

    fire = config.get('fire_smoke', {})
    fire_path = Path(fire.get('validation_file') or ROOT/'data/fire_smoke_validation.json')
    fire_validation = mapping(load(fire_path))
    fire_samples = rows(fire_validation.get('samples'))
    positive_samples = [sample for sample in fire_samples
                        if mapping(sample.get('case')).get('positive') is True]
    positive_detected = sum(any(d.get('label') in {'fire', 'smoke'}
                                for d in rows(mapping(sample.get('result')).get('detections')))
                            for sample in positive_samples)

    model = Path(config.get('model_path', ''))
    benchmark = load(model.parent/'benchmark.json') if model.name else None
    benchmark = [item for item in rows(benchmark) if item.get('model_filename') == str(model)]
    prompt = next((item for item in benchmark if item.get('n_prompt', 0) and item.get('n_gen') == 0), None)
    generation = next((item for item in benchmark if item.get('n_gen', 0) and item.get('n_prompt') == 0), None)
    monitor_path = ROOT/'data/monitor/photo-suite-smoke-test.json'
    monitor_test = mapping(load(monitor_path))
    monitor_events = rows(monitor_test.get('events'))
    real_video_path = ROOT/'data/monitor/sessions/dd8bcc2d6c234a9096084f4032a28178/test-summary.json'
    real_video = mapping(load(real_video_path))
    memory = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in {'MemTotal', 'MemAvailable'}:
                memory[key] = int(value.split()[0]) / 1024**2
    except (OSError, ValueError):
        pass
    try:
        load_average = os.getloadavg()[0]
    except OSError:
        load_average = None
    latest_agent = None
    for report_path in sorted(agent.REPORTS.glob('*.json'), reverse=True)[:40]:
        report = mapping(load(report_path))
        execution = mapping(report.get('agent_execution'))
        elapsed = number(report.get('elapsed_seconds'))
        if (execution.get('version') == 'local-tool-agent-v1' and elapsed is not None
                and execution.get('status') in {'completed', 'fallback'}):
            latest_agent = {'elapsed_seconds': elapsed,
                            'status': execution['status'],
                            'rounds': number(execution.get('rounds')),
                            'at': report.get('created_at')}
            break
    tools = ['人员检测', '烟火检测', '本地知识检索', '预警评估']
    return {
        'generated_at': agent.utc_now(),
        'platform': {'name': cpu_name, 'architecture': platform.machine(),
                     'npu_available': Path('/dev/aipu').exists()},
        'resources': {'logical_cores': os.cpu_count(), 'load_average_1m': load_average,
                      'memory_total_gib': memory.get('MemTotal'),
                      'memory_available_gib': memory.get('MemAvailable'),
                      'scope': '整机当前状态；1分钟负载不是CPU使用率，也不是模型峰值资源占用'},
        'agent': {'enabled': config.get('agent', {}).get('enabled', False),
                  'max_rounds': max(1, min(3, config.get('agent', {}).get('max_rounds', 3))),
                  'tools': tools, 'memory': '按图片哈希与模型指纹复用成功结果',
                  'recovery': '检测失败保留为未知；证据不符时规则兜底'},
        'models': {
            'person_backend': person.get('backend', 'unknown'),
            'person_device': 'CPU · PyTorch' if person.get('backend') == 'damoyolo'
                             else ('CIX NPU配置 · 设备已发现' if Path('/dev/aipu').exists()
                                   else 'CIX NPU配置 · 设备未接通') if person.get('backend') == 'cix_yolov8s'
                             else ('NPU配置 · 设备已发现' if Path('/dev/aipu').exists()
                                   else 'NPU配置 · 未发现设备'),
            'fire_device': 'CPU · PyTorch',
            'llm_device': ('CPU · llama.cpp' if benchmark and benchmark[0].get('backends') == 'CPU'
                           else benchmark[0].get('backends', 'llama.cpp · 本地') if benchmark
                           else ('CPU · llama.cpp' if 'cpu' in config.get('llama_cli', '').lower()
                                 else 'llama.cpp · 本地'))},
        'measurements': {
            'llm_prompt_tokens_per_second': number(prompt.get('avg_ts')) if prompt else None,
            'llm_prompt_samples': len(prompt['samples_ts']) if prompt and isinstance(prompt.get('samples_ts'), list) else 0,
            'llm_generation_tokens_per_second': number(generation.get('avg_ts')) if generation else None,
            'llm_generation_samples': len(generation['samples_ts']) if generation and isinstance(generation.get('samples_ts'), list) else 0,
            'llm_benchmark_at': generation.get('test_time') if generation else None,
            'llm_benchmark_threads': generation.get('n_threads') if generation else None,
            'person_sample_count': len(person_results),
            'person_timed_samples': len(person_times),
            'person_mean_seconds': (sum(person_times)/len(person_times) if person_times else None),
            'fire_sample_count': len(fire_samples),
            'fire_positive_count': len(positive_samples),
            'fire_positive_detected': positive_detected,
            'monitor_status': monitor_test.get('status'),
            'monitor_frames': number(monitor_test.get('frames_processed')),
            'monitor_events': len(monitor_events),
            'monitor_error': monitor_test.get('error'),
            'real_video_status': real_video.get('status'),
            'real_video_frames': number(real_video.get('frames_processed')),
            'real_video_duration_seconds': number(mapping(real_video.get('source')).get('duration_seconds')),
            'real_video_last_cycle_seconds': number(real_video.get('cycle_seconds')),
            'latest_agent_elapsed_seconds': latest_agent.get('elapsed_seconds') if latest_agent else None,
            'latest_agent_status': latest_agent.get('status') if latest_agent else None,
            'latest_agent_rounds': latest_agent.get('rounds') if latest_agent else None,
            'latest_agent_at': latest_agent.get('at') if latest_agent else None,
            'latest_agent_sample_count': 1 if latest_agent else 0},
        'sources': {'person': 'data/' + person_path.name,
                    'fire_smoke': 'data/' + fire_path.name,
                    'llm': model.parent.name + '/benchmark.json' if model.name else '未配置',
                    'photo_video': 'data/monitor/photo-suite-smoke-test.json',
                    'real_video': 'data/monitor/sessions/本机验证记录'},
        'scope': ['以上为历史验证记录，修改模型、配置或硬件后需重新测试。',
                  '人员样例仅证明推理输出与耗时，count字段不是独立人工真值，不计算准确率。',
                  '烟火正样例统计任意烟/火检出，不等于框级召回率或现场准确率。',
                  '照片序列仅验证流程；实拍片段是人员与烟火负例，没有逐帧人数真值。',
                  'Tokens/s为模型基准，不是完整Agent响应速度；单图耗时不含模型加载。']
    }


def validate_image(data):
    name = str(data.get('image_name', ''))
    encoded = data.get('image_base64')
    if not isinstance(encoded, str):
        raise ValueError('请先选择一张 JPG 或 PNG 图片')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('图片编码无效') from exc
    if not raw or len(raw) > 10 * 1024 * 1024:
        raise ValueError('图片不能为空且不能超过 10 MB')
    if raw.startswith(b'\xff\xd8\xff'):
        suffix = '.jpg'
    elif raw.startswith(b'\x89PNG\r\n\x1a\n'):
        suffix = '.png'
    else:
        raise ValueError('只支持 JPG、PNG 图片')
    return raw, suffix, name


def stop_job_process(process):
    """Stop only the private process group created for this request."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit before a nested detector or llama.cpp child does.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)


def run_job_process(job_id, command, log):
    with LOCK:
        if read_json(JOBS/f'{job_id}.json')['status'] == 'cancelled':
            return None
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        RUNNING_JOBS[job_id] = process
    try:
        return subprocess.CompletedProcess(command, process.wait(timeout=JOB_TIMEOUT))
    except BaseException:
        stop_job_process(process)
        raise
    finally:
        with LOCK:
            RUNNING_JOBS.pop(job_id, None)


def cancel_job(job_id):
    if not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise ValueError('任务编号无效')
    with LOCK:
        path = JOBS/f'{job_id}.json'
        state = read_json(path)
        if state['status'] not in {'queued', 'running'}:
            return state
        state.update(status='cancelled', finished_at=agent.utc_now(), error='本轮已停止，可以重新提问。')
        write_json(path, state)
        process = RUNNING_JOBS.get(job_id)
    if process is not None:
        stop_job_process(process)
    return state


def worker(job_id, request, image):
    path = JOBS / f'{job_id}.json'
    result_file = JOBS / f'{job_id}.result'
    state = None
    upload = image
    context_file = JOBS / f'{job_id}.context'
    progress_file = JOBS / f'{job_id}.progress'
    try:
        with LOCK:
            state = read_json(path)
            if state['status'] != 'queued':
                return
            state.update(status='running', started_at=agent.utc_now())
            write_json(path, state)
        session = None
        context = {}
        if request.get('agent'):
            session = read_json(conversation_path(request['conversation_id']))
            context, image = conversation_context(session, image, request.get('clear_scene', False))
            write_json(context_file, context)
        observation = request.get('observation', '')
        if context.get('scene_reused') and context.get('previous_observation'):
            observation = (context['previous_observation'][:2000] + ('\n本轮补充：' + observation if observation else ''))
        command = [sys.executable, '-u', str(ROOT / 'agent.py'), 'run', '--task', request['task'],
                   '--observation', observation,
                   '--result-file', str(result_file)]
        if request.get('agent'):
            command.extend(['--agent', '--agent-context', str(context_file), '--progress-file', str(progress_file)])
        if image:
            command.extend(['--image', str(image)])
        if image and request.get('detect', True):
            command.append('--npu-detect')
        if request.get('llm', True):
            command.append('--llm')
        if request.get('safety_context'):
            command.extend(['--safety-context', json.dumps(request['safety_context'], ensure_ascii=False)])
        with (JOBS / f'{job_id}.log').open('w', encoding='utf-8') as log:
            completed = run_job_process(job_id, command, log)
        if completed is None:
            return
        if completed.returncode or not result_file.is_file():
            raise RuntimeError('巡查执行失败，日志：' + (JOBS / f'{job_id}.log').read_text()[-1600:])
        record = read_json(result_file)
        with LOCK:
            if read_json(path)['status'] == 'cancelled':
                return
            if session is not None:
                session['case_ids'] = (session.get('case_ids', []) + [record['case_id']])[-24:]
                session['scene_case_id'] = record['case_id'] if record.get('image') else None
                session['updated_at'] = agent.utc_now()
                write_json(conversation_path(session['id']), session)
            state.update(status='done', case_id=record['case_id'], finished_at=agent.utc_now())
            write_json(path, state)
    except Exception as exc:
        # Every worker exception must terminate the job, including schema bugs.
        with (JOBS/f'{job_id}.log').open('a', encoding='utf-8') as log:
            traceback.print_exc(file=log)
        message = ('本轮处理超过 15 分钟，已停止。请减少问题范围后重试。'
                   if isinstance(exc, subprocess.TimeoutExpired) else f'处理失败（{type(exc).__name__}）：{exc}')
        if state is not None:
            state.update(status='failed', error=message[-1800:], finished_at=agent.utc_now())
    finally:
        if state is not None:
            with LOCK:
                current = read_json(path)
                if current['status'] == 'cancelled':
                    state = current
                write_json(path, state)
        result_file.unlink(missing_ok=True)
        context_file.unlink(missing_ok=True)
        if upload:
            upload.unlink(missing_ok=True)


def submit(request):
    if not isinstance(request.get('task'), str) or not request['task'].strip() or len(request['task']) > 1000:
        raise ValueError('请输入 1 到 1000 字的巡查任务')
    if not isinstance(request.get('observation', ''), str) or len(request.get('observation', '')) > 4000:
        raise ValueError('现场观察不能超过 4000 字')
    for field in ('detect', 'llm', 'clear_scene'):
        if field in request and not isinstance(request[field], bool):
            raise ValueError('选项格式错误')
    triage.validate_context(request.get('safety_context', {}))
    uploaded = 'image_base64' in request
    raw, suffix, name = validate_image(request) if uploaded else (None, None, '')
    use_agent = request.get('llm', True) and read_json(ROOT/'config.json').get('agent', {}).get('enabled', False)
    session = None
    new_session = False
    if use_agent:
        conversation_id = request.get('conversation_id')
        if conversation_id:
            session = read_json(conversation_path(conversation_id))
        else:
            new_session = True
            conversation_id = uuid.uuid4().hex
            session = {'id': conversation_id, 'case_ids': [], 'scene_case_id': None,
                       'created_at': agent.utc_now()}
        parent = request.get('parent_case_id')
        if parent:
            previous = read_json(record_path(parent))
            if not session['case_ids'] or session['case_ids'][-1] != parent:
                # Following a historical record starts a branch at that record.
                conversation_id = uuid.uuid4().hex
                session = {'id': conversation_id, 'case_ids': [parent],
                           'scene_case_id': parent if previous.get('image') else None,
                           'created_at': agent.utc_now()}
                new_session = True
    with LOCK:
        pending = sum(read_json(p).get('status') in {'queued', 'running'} for p in JOBS.glob('*.json'))
        if pending >= 3:
            raise ValueError('已有任务执行或排队，请稍后再试')
        job_id = uuid.uuid4().hex
        if new_session:
            write_json(conversation_path(session['id']), session)
        image = None
        if uploaded:
            UPLOADS.mkdir(parents=True, exist_ok=True)
            image = UPLOADS / f'{job_id}{suffix}'
            image.write_bytes(raw)
        write_json(JOBS / f'{job_id}.json', {'id': job_id, 'status': 'queued', 'image_name': name,
                                          'conversation_id': session['id'] if session else None,
                                          'task': request['task'], 'created_at': agent.utc_now()})
        payload = {k: request[k] for k in ('task', 'observation', 'detect', 'llm', 'safety_context', 'clear_scene') if k in request}
        if session is not None:
            payload.update(agent=True, conversation_id=session['id'])
        POOL.submit(worker, job_id, payload, image)
    return {'id': job_id, 'conversation_id': session['id'] if session else None}


def batch_folder(batch_id):
    if not re.fullmatch(r'\d{8}-\d{6}', batch_id):
        raise ValueError('批次编号无效')
    return BATCHES / batch_id


def list_batches():
    return [read_json(p) for p in sorted(BATCHES.glob('*/status.json'), reverse=True)[:20]]


def batch_details(batch_id, query):
    folder = batch_folder(batch_id)
    status = read_json(folder/'status.json')
    offset = max(0, int(query.get('offset', ['0'])[0]))
    limit = max(1, min(100, int(query.get('limit', ['30'])[0])))
    level = query.get('level', [''])[0]
    if level not in {'', '0', '1', '2', '3', 'unknown', 'missed'}:
        raise ValueError('分级筛选无效')
    items, count = [], 0
    if (folder/'results.jsonl').exists():
        with (folder/'results.jsonl').open() as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # The worker may currently be appending the last line.
                if level == 'unknown' and entry.get('score') is not None:
                    continue
                if level == 'missed' and not any(v.get('fn', 0) for v in entry.get('evaluation', {}).values()):
                    continue
                if level in {'0', '1', '2', '3'} and str(entry.get('level')) != level:
                    continue
                if offset <= count < offset + limit:
                    items.append({key: entry.get(key) for key in
                                  ('index', 'name', 'split', 'score', 'level', 'label', 'case_id', 'error', 'annotation_error', 'evaluation')})
                count += 1
    return {'status': status, 'items': items, 'matched': count, 'offset': offset, 'limit': limit}


def batch_worker(batch_id):
    folder = batch_folder(batch_id)
    try:
        with (folder/'worker.log').open('a') as log:
            result = subprocess.run([agent.CONFIG['fire_smoke']['python'], '-u',
                str(ROOT/'tools/batch_fire_smoke.py'), '--batch-id', batch_id],
                cwd=ROOT, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError('批量任务失败，请查看该批次 worker.log')
    except (OSError, RuntimeError) as exc:
        status = read_json(folder/'status.json')
        status.update(status='failed', error=str(exc))
        write_json(folder/'status.json', status)


def submit_batch():
    with LOCK:
        if any(s.get('status') in {'queued', 'preparing', 'running'} for s in list_batches()):
            raise ValueError('已有数据集批量测试运行中，请查看进度，避免重复启动')
        batch_id = __import__('datetime').datetime.now().strftime('%Y%m%d-%H%M%S')
        write_json(batch_folder(batch_id)/'status.json',
                   {'id': batch_id, 'status': 'queued', 'completed': 0, 'total': None, 'errors': 0})
        POOL.submit(batch_worker, batch_id)
    return {'id': batch_id}


def record_path(case_id):
    if not isinstance(case_id, str) or not re.fullmatch(r'\d{8}-\d{6}-[a-f0-9]{6}', case_id):
        raise ValueError('记录编号无效')
    return agent.REPORTS / f'{case_id}.json'


def confirm(case_id, text):
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError('请输入真实的人工复核记录，不超过 2000 字')
    with LOCK:
        record = agent.with_task_result(read_json(record_path(case_id)))
        record['human_confirmation'] = text.strip()
        record.setdefault('confirmation_history', []).append({'text': text.strip(), 'at': agent.utc_now()})
        record.setdefault('events', []).append({'step': 'human_confirmation', 'status': 'recorded', 'at': agent.utc_now()})
        agent.save_record(record)
    return record


def import_knowledge(request):
    title, body = request.get('title'), request.get('body')
    if not isinstance(title, str) or not title.strip() or len(title) > 200:
        raise ValueError('请填写文档名称、来源和版本（不超过 200 字）')
    if not isinstance(body, str) or not body.strip() or len(body) > 100000:
        raise ValueError('预案文本不能为空且不能超过 10 万字')
    directory = ROOT / 'data' / 'knowledge_sources'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{uuid.uuid4().hex}.txt'
    path.write_text(body, encoding='utf-8')
    agent.add_doc(argparse.Namespace(file=str(path), title=title.strip()))
    return {'ok': True}


class Handler(BaseHTTPRequestHandler):
    def json(self, value, status=200):
        raw = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(raw)

    def send_file(self, path):
        if not path.is_file():
            self.send_error(404)
            return
        raw = path.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        try:
            if path == '/api/status':
                self.json(capabilities())
            elif path == '/api/evaluation':
                self.json(evaluation_snapshot(read_json(ROOT/'config.json')))
            elif path == '/api/app-info':
                self.json({'application': 'cn.yingjixianxing.Desktop', 'protocol': 1,
                           'project_root': str(ROOT.resolve())})
            elif path == '/api/monitor':
                self.json(MONITOR.status())
            elif re.fullmatch(r'/api/monitor/assets/[a-f0-9]{32}/(?:frames/frame-\d+|events/[a-f0-9]{32})\.jpg', path):
                session, kind, filename = path.split('/')[-3:]
                self.send_file(MONITOR.asset(session, kind, filename))
            elif path == '/api/batches':
                self.json(list_batches())
            elif re.fullmatch(r'/api/batches/\d{8}-\d{6}', path):
                self.json(batch_details(path.rsplit('/', 1)[1], parse_qs(urlsplit(self.path).query)))
            elif re.fullmatch(r'/api/batches/\d{8}-\d{6}/export', path):
                self.send_file(batch_folder(path.split('/')[3])/'results.csv')
            elif path == '/api/history':
                items = []
                for p in sorted(agent.REPORTS.glob('*.json'), reverse=True):
                    record = read_json(p)
                    if record.get('dataset_sample'):
                        continue
                    items.append({k: record.get(k) for k in ('case_id', 'task', 'created_at', 'risk_level', 'human_confirmation')})
                    if len(items) >= 100:
                        break
                self.json(items)
            elif re.fullmatch(r'/api/jobs/[a-f0-9]{32}', path):
                state = read_json(JOBS / f'{path.rsplit("/", 1)[1]}.json')
                if state.get('started_at') or state.get('created_at'):
                    started = datetime.fromisoformat(state.get('started_at') or state['created_at'])
                    state['elapsed_seconds'] = max(0, int((datetime.now(timezone.utc)-started).total_seconds()))
                progress = JOBS / f'{state["id"]}.progress'
                if progress.is_file():
                    state['progress'] = read_json(progress)
                if state.get('case_id'):
                    state['record'] = agent.with_task_result(read_json(record_path(state['case_id'])))
                self.json(state)
            elif re.fullmatch(r'/api/conversations/[a-f0-9]{32}', path):
                self.json(conversation_view(path.rsplit('/', 1)[1]))
            elif path.startswith('/api/records/'):
                self.json(agent.with_task_result(read_json(record_path(path.rsplit('/', 1)[1]))))
            elif path == '/':
                self.send_file(WEB / 'index.html')
            elif path in {'/app.js', '/style.css', '/monitor.js', '/desktop.js', '/desktop.css'}:
                self.send_file(WEB / path[1:])
            elif path in {'/demo/fire.jpg', '/demo/people.jpg'}:
                self.send_file(ROOT / ('samples/fire-smoke/ESE_fire_10.jpg' if path == '/demo/fire.jpg'
                                      else 'samples/000000559842.jpg'))
            elif path in {'/docs/决赛评分标准对照与网页优化.md', '/docs/技术白皮书与PPT素材.md',
                          '/docs/CIX迁移说明.md', '/docs/5分钟Demo演讲稿.md'}:
                self.send_file(ROOT / path[1:])
            elif re.fullmatch(r'/data/batches/\d{8}-\d{6}/(?:summary\.(?:html|md|json)|results\.csv)', path):
                self.send_file(ROOT/path[1:])
            elif path.startswith('/data/'):
                parts = Path(path).parts
                if len(parts) != 4 or parts[2] not in {'reports', 'evidence'}:
                    raise ValueError('文件地址无效')
                target = (ROOT / path[1:]).resolve()
                if not target.is_relative_to((ROOT / 'data' / parts[2]).resolve()):
                    raise ValueError('文件地址无效')
                self.send_file(target)
            else:
                self.send_error(404)
        except FileNotFoundError:
            self.json({'error': '记录不存在'}, 404)
        except (ValueError, OSError, sqlite3.Error) as exc:
            self.json({'error': str(exc)}, 400)

    def do_POST(self):
        # Only same-origin browser requests may alter local records.
        origin = self.headers.get('Origin')
        host = self.headers.get('Host', '')
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        # On a wildcard listener, the accepted socket identifies the actual
        # local interface used by this request. Do not trust arbitrary Host values.
        connection = getattr(self, 'connection', None)
        if connection is not None:
            local_address = connection.getsockname()[0]
            allowed.add(f'{local_address}:{self.server.server_port}')
        if host not in allowed or (origin and origin != f'http://{host}'):
            self.json({'error': '请从板卡 IP 地址或本机地址打开页面后提交请求'}, 403)
            return
        try:
            size = int(self.headers.get('Content-Length', 0))
            path = urlsplit(self.path).path
            if path == '/api/monitor/video':
                if not 0 < size <= monitor.MAX_VIDEO_BYTES:
                    raise ValueError('视频不能为空且不能超过 200 MB')
                name = unquote(self.headers.get('X-File-Name', ''))
                self.json(MONITOR.upload(self.rfile, size, name), 201)
                return
            if not 0 < size <= MAX_BODY:
                raise ValueError('请求过大或内容为空')
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError('请求格式错误')
            if path == '/api/run':
                self.json(submit(request), 202)
            elif re.fullmatch(r'/api/jobs/[a-f0-9]{32}/cancel', path):
                self.json(cancel_job(path.split('/')[3]))
            elif path == '/api/monitor/start':
                self.json(MONITOR.start(request), 202)
            elif path == '/api/monitor/stop':
                self.json(MONITOR.stop())
            elif path == '/api/batches':
                self.json(submit_batch(), 202)
            elif path == '/api/confirm':
                self.json(confirm(request.get('case_id'), request.get('text')))
            elif path == '/api/knowledge':
                self.json(import_knowledge(request))
            else:
                self.send_error(404)
        except FileNotFoundError:
            self.json({'error': '记录不存在'}, 404)
        except (ValueError, OSError, sqlite3.Error) as exc:
            self.json({'error': str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description='断网应急巡查本地网页')
    parser.add_argument('--host', default='127.0.0.1',
                        help='监听地址；使用 0.0.0.0 允许同一局域网内设备访问')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--ready-file', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    agent.connect().close()
    JOBS.mkdir(parents=True, exist_ok=True)
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        parser.exit(1, f'本地网页启动失败：{exc}\n请在板卡本机终端运行，或使用 --port 更换端口。\n')
    for p in JOBS.glob('*.json'):
        record = read_json(p)
        if record.get('status') in {'queued', 'running'}:
            record.update(status='failed', error='上次服务停止，任务已中断，请重新提交')
            write_json(p, record)
    access_host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
    actual_port = server.server_port
    if args.ready_file:
        import os
        write_json(args.ready_file, {'host': args.host, 'port': actual_port, 'pid': os.getpid()})
    print(f'网页：http://{access_host}:{actual_port}  （Ctrl+C 停止）', flush=True)
    if args.host == '0.0.0.0':
        print(f'局域网访问：http://<板卡的局域网IP>:{args.port}；在板卡终端运行 hostname -I 查看 IP。', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Requests own process groups so cancellation can stop nested inference.
        # Stop them explicitly before the service/desktop exits.
        for path in JOBS.glob('*.json'):
            if read_json(path).get('status') in {'queued', 'running'}:
                cancel_job(path.stem)
        MONITOR.stop()
        server.server_close()
        POOL.shutdown(wait=True)


if __name__ == '__main__':
    main()
