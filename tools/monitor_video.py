#!/usr/bin/env python3
"""Sample a local video in wall-clock time; keep detection models resident."""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monitor import AlarmTracker, now, read_json, write_json


def parent_lifetime(parent_pid):
    # Stop descendants if the web server or supervisor disappears unexpectedly.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), '无法设置监测进程生命周期')
    if os.getppid() != parent_pid:
        raise RuntimeError('启动进程已退出')


def load_detector(kind, settings):
    if kind == 'person' and settings.get('backend') == 'cix_yolov8s':
        from tools.detect_yolov8_npu import LocalDetector
        return LocalDetector(settings['model_path'], settings.get('threshold', .5),
                             settings.get('nms_threshold', .45))
    if kind == 'person':
        if settings.get('backend', 'damoyolo') != 'damoyolo':
            raise ValueError('视频监测不支持该人员检测后端')
        from tools.detect_person import LocalDetector
    elif kind == 'fire_smoke':
        from tools.detect_fire_smoke import LocalDetector
    else:
        raise ValueError('未知检测工具')
    return LocalDetector(settings['model_dir'], settings.get('threshold', .5 if kind == 'person' else .6),
                         settings.get('max_side', 640), settings.get('threads', 8))


def detector_process(kind, settings, parent_pid):
    detector = None
    def send(value):
        print(json.dumps(value, ensure_ascii=False), flush=True)
    try:
        parent_lifetime(parent_pid)
        detector = load_detector(kind, settings)
        send({'ready': True})
        for line in sys.stdin:
            image = json.loads(line)
            if image is None:
                break
            send({'result': detector.infer(image)})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            send({'error': str(exc)})
        except (BrokenPipeError, OSError):
            pass
    finally:
        if detector is not None and hasattr(detector, 'close'):
            detector.close()


class Detector:
    def __init__(self, kind, settings, stop):
        self.stop = stop
        # CPU PyTorch and CIX NOE are installed in separate Python environments.
        # Launch each interpreter directly so site-packages never cross over.
        self.process = subprocess.Popen([
            settings.get('python', sys.executable), '-u', str(Path(__file__).resolve()),
            '--detector', kind, '--detector-settings', json.dumps(settings),
            '--parent-pid', str(os.getpid())
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def send(self, image):
        try:
            self.process.stdin.write(json.dumps(str(image)) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError('检测进程意外退出') from exc

    def receive(self, timeout=90):
        deadline = time.monotonic() + timeout
        while not self.stop.is_set():
            if self.selector.select(.1):
                try:
                    line = self.process.stdout.readline()
                    if not line:
                        raise EOFError
                    value = json.loads(line)
                except (EOFError, ValueError) as exc:
                    raise RuntimeError('检测进程意外退出') from exc
                if 'error' in value:
                    raise RuntimeError(value['error'])
                return value
            if self.process.poll() is not None:
                raise RuntimeError('检测进程意外退出')
            if time.monotonic() >= deadline:
                raise RuntimeError('本地检测模型响应超时')
        raise InterruptedError('监测已停止')

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('null\n')
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()


def capture(video, position, output):
    result = subprocess.run([
        'ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
        '-protocol_whitelist', 'file,pipe', '-format_whitelist', 'mov,matroska,webm,avi',
        '-ss', f'{position:.3f}', '-i', str(video),
        '-frames:v', '1', '-vf', "scale=960:960:force_original_aspect_ratio=decrease:force_divisible_by=2",
        '-threads', '1', '-q:v', '3', '-y', str(output)
    ], capture_output=True, timeout=12)
    if result.returncode or not output.is_file() or not output.stat().st_size:
        raise RuntimeError('视频取帧失败：' + result.stderr.decode(errors='replace')[-500:])


def run(folder, parent_pid):
    loaded_at = time.monotonic()
    parent_lifetime(parent_pid)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    settings = read_json(folder / 'settings.json')
    state = read_json(folder / 'status.json')
    session = folder.name
    for name in ('frames', 'events'):
        (folder / name).mkdir(exist_ok=True)
    config = settings['config']
    affinity = config.get('cpu_affinity', '')
    if affinity and hasattr(os, 'sched_setaffinity'):
        cores = set()
        for part in affinity.split(','):
            ends = [int(n) for n in part.split('-')]
            cores.update(range(ends[0], ends[-1] + 1))
        available = cores & os.sched_getaffinity(0)
        if available:
            os.sched_setaffinity(0, available)
    tracker = AlarmTracker(settings['detect_people'] and settings['person_alert'])
    events = {}
    detectors = {}
    last_frame = None

    def save_events(changes, frame=None):
        for event in changes:
            previous = events.get(event['id'])
            # Preserve the initial evidence; replace it only on an escalation.
            needs_image = previous is None or previous.get('stage') != event['stage']
            event['image_url'] = f'/api/monitor/assets/{session}/events/{event["id"]}.jpg'
            event['image_size'] = state.get('image_size') if needs_image else previous.get('image_size')
            event['detections'] = state.get('detections', []) if needs_image else previous.get('detections', [])
            if frame and needs_image:
                shutil.copyfile(frame, folder / 'events' / (event['id'] + '.jpg'))
            events[event['id']] = event
            write_json(folder / 'events' / (event['id'] + '.json'), event)
        while len(events) > 200:
            oldest = next(iter(events))
            events.pop(oldest)
            for suffix in ('.json', '.jpg'):
                (folder / 'events' / (oldest + suffix)).unlink(missing_ok=True)
        state['events'] = list(reversed(list(events.values())))

    def persist():
        state['updated_at'] = now()
        write_json(folder / 'status.json', state)

    try:
        # Each detector has its own resident process and CPU runtime settings.
        detectors['fire_smoke'] = Detector('fire_smoke', config['fire_smoke'], stop)
        if settings['detect_people']:
            detectors['person'] = Detector('person', config['person_detection'], stop)
        for detector in detectors.values():
            detector.receive()
        if stop.is_set():
            raise InterruptedError('监测已停止')
        duration = settings['source']['duration_seconds']
        interval = settings['interval_seconds']
        started = time.monotonic()
        next_due = started
        loop_number = 0
        state.update(status='running', message='正在按视频时间持续取帧检测', model_load_seconds=round(started - loaded_at, 3))
        persist()
        while not stop.is_set():
            if stop.wait(max(0, next_due - time.monotonic())):
                break
            elapsed = time.monotonic() - started
            if not settings['loop'] and elapsed >= duration:
                state.update(status='completed', message='视频回放结束')
                break
            current_loop = int(elapsed // duration)
            if current_loop != loop_number:
                save_events(tracker.end(now(), '视频重新回放，重新累计检测次数'), last_frame)
                loop_number = current_loop
            position = elapsed % duration
            seq = state['frames_processed'] + 1
            frame = folder / 'frames' / f'frame-{seq}.jpg'
            cycle_started = time.monotonic()
            capture(settings['source_path'], position, frame)
            if stop.is_set():
                break
            for detector in detectors.values():
                detector.send(frame)
            results = {kind: detector.receive(timeout=45)['result'] for kind, detector in detectors.items()}
            detections = [d for result in results.values() for d in result['detections']]
            state.update(frames_processed=seq, position_seconds=round(position, 2), replay_count=loop_number,
                         sampled_at=now(), detections=detections, image_size=results['fire_smoke']['image_size'],
                         frame_url=f'/api/monitor/assets/{session}/frames/{frame.name}',
                         people_count=sum(d['label'] == 'person' for d in detections) if settings['detect_people'] else None,
                         inference_seconds={k: r['elapsed_seconds'] for k, r in results.items()},
                         cycle_seconds=round(time.monotonic() - cycle_started, 3),
                         model_info={k: {name: r.get(name) for name in ('source', 'model_sha256', 'device')} for k, r in results.items()})
            save_events(tracker.update(detections, round(position, 2), now()), frame)
            last_frame = frame
            persist()
            # Only retain recent normal frames, never an unbounded frame queue.
            for old in (folder / 'frames').glob('frame-*.jpg'):
                if int(old.stem.split('-')[1]) < seq - 2:
                    old.unlink(missing_ok=True)
            next_due = max(next_due + interval, time.monotonic())
    except InterruptedError:
        pass
    except Exception as exc:
        state.update(status='failed', error=str(exc), message='监测故障，请检查后重新启动')
    finally:
        if state['status'] in {'starting', 'running', 'stopping'}:
            state.update(status='stopped', message='监测已停止')
        save_events(tracker.end(now(), '监测结束，后续画面未检测'), last_frame)
        for detector in detectors.values():
            detector.close()
        persist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', type=Path)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--detector', choices=['person', 'fire_smoke'])
    parser.add_argument('--detector-settings')
    args = parser.parse_args()
    if args.detector:
        if not args.detector_settings:
            parser.error('缺少检测模型配置')
        detector_process(args.detector, json.loads(args.detector_settings), args.parent_pid)
        sys.exit(0)
    if args.session_dir is None:
        parser.error('缺少监测任务目录')
    # Prevent two web instances from running competing video monitors.
    lock_path = args.session_dir.parents[1] / 'worker.lock'
    with lock_path.open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state = read_json(args.session_dir / 'status.json')
            state.update(status='failed', error='另一个网页服务已有监测任务运行', updated_at=now())
            write_json(args.session_dir / 'status.json', state)
            sys.exit(1)
        run(args.session_dir, args.parent_pid)
