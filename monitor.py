"""Local video sources and lifecycle for one background monitoring process."""
import datetime as dt
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import uuid

MAX_VIDEO_BYTES = 200 * 1024 * 1024
ACTIVE_STATES = {'starting', 'running', 'stopping'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.webm', '.avi'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def probe_video(path):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
            '-format_whitelist', 'mov,matroska,webm,avi',
            '-select_streams', 'v:0', '-show_entries',
            'format=duration:stream=width,height,duration', '-of', 'json', str(path)
        ], capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('视频检查超时，请尝试较短的视频或转换为 MP4') from exc
    if result.returncode:
        raise ValueError('无法读取视频，请选择有效的 MP4、MOV、MKV、WebM 或 AVI 文件')
    info = json.loads(result.stdout)
    streams = info.get('streams', [])
    if not streams:
        raise ValueError('文件中没有可解码的视频画面')
    duration = float(info.get('format', {}).get('duration') or streams[0].get('duration') or 0)
    width, height = int(streams[0].get('width', 0)), int(streams[0].get('height', 0))
    if not math.isfinite(duration) or duration <= 0 or not (0 < width <= 8192 and 0 < height <= 8192):
        raise ValueError('视频时长或分辨率无效，最长边需不超过 8192 像素')
    return {'duration_seconds': round(duration, 3), 'width': width, 'height': height}


class AlarmTracker:
    """Notify on first detection, escalate repeated hits, merge an ongoing event."""
    def __init__(self, people=False):
        self.labels = ('fire', 'smoke', 'person') if people else ('fire', 'smoke')
        self.states = {label: {'hits': 0, 'misses': 0, 'event': None} for label in self.labels}

    def update(self, detections, position, timestamp):
        changed = []
        for label, state in self.states.items():
            items = [d for d in detections if d['label'] == label]
            if items:
                state['hits'] += 1
                state['misses'] = 0
                event = state['event']
                if event is None:
                    event = {'id': uuid.uuid4().hex, 'label': label, 'status': 'active',
                             'stage': 'first_seen', 'first_seen': timestamp,
                             'first_position_seconds': position, 'samples': 0}
                    state['event'] = event
                event.update(last_seen=timestamp, last_position_seconds=position,
                             count=len(items), confidence=max(d['confidence'] for d in items))
                event['samples'] += 1
                if state['hits'] >= 2:
                    event['stage'] = 'persistent'
                event['level'] = ('critical' if label == 'fire' and event['stage'] == 'persistent'
                                  else 'notice' if label == 'person' else 'warning')
                changed.append(dict(event))
            else:
                state['hits'] = 0
                state['misses'] += 1
                if state['event'] and state['misses'] >= 3:
                    event = state['event']
                    event.update(status='ended', ended_at=timestamp, end_reason='连续三次未检出；不表示现场安全')
                    changed.append(dict(event))
                    state['event'] = None
        return changed

    def end(self, timestamp, reason):
        changed = []
        for state in self.states.values():
            if state['event']:
                event = state['event']
                event.update(status='ended', ended_at=timestamp, end_reason=reason)
                changed.append(dict(event))
            state.update(hits=0, misses=0, event=None)
        return changed


class Controller:
    def __init__(self, root):
        self.root = Path(root)
        self.data = self.root / 'data' / 'monitor'
        self.process = None
        self.session_id = None
        self.log = None
        self.lock = threading.RLock()

    def source(self, source_id):
        if not isinstance(source_id, str) or not re.fullmatch(r'(?:demo|[a-f0-9]{32})', source_id):
            raise ValueError('请选择已导入的视频')
        item = read_json(self.data / 'videos' / f'{source_id}.json')
        path = (self.data / 'videos' / item['filename']).resolve()
        if not path.is_relative_to((self.data / 'videos').resolve()) or not path.is_file():
            raise ValueError('视频文件已不可用，请重新导入')
        return item, path

    def sources(self):
        result = []
        for path in sorted((self.data / 'videos').glob('*.json')):
            try:
                item, _ = self.source(path.stem)
                result.append({k: v for k, v in item.items() if k != 'filename'})
            except (OSError, ValueError, KeyError):
                continue
        return result

    def upload(self, stream, size, name):
        if not 0 < size <= MAX_VIDEO_BYTES:
            raise ValueError('视频不能为空且不能超过 200 MB')
        name = Path(str(name).replace('\\', '/')).name[:160]
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise ValueError('支持 MP4、MOV、MKV、WebM、AVI 视频')
        source_id = uuid.uuid4().hex
        directory = self.data / 'videos'
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (source_id + suffix)
        try:
            remaining = size
            with target.open('xb') as output:
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError('视频传输中断，请重新导入')
                    output.write(block)
                    remaining -= len(block)
            info = probe_video(target)
            item = dict(info, id=source_id, name=name, filename=target.name,
                        kind='video', size_bytes=size, imported_at=now())
            write_json(directory / (source_id + '.json'), item)
            return {k: v for k, v in item.items() if k != 'filename'}
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def _session(self):
        if self.session_id:
            return self.session_id
        try:
            value = read_json(self.data / 'current.json')['session_id']
            if re.fullmatch(r'[a-f0-9]{32}', value):
                return value
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def status(self):
        with self.lock:
            session = self._session()
            state = read_json(self.data / 'sessions' / session / 'status.json') if session else {'status': 'idle'}
            alive = self.process is not None and self.process.poll() is None
            if state.get('status') in ACTIVE_STATES and not alive:
                state.update(status='failed', error='监测进程已停止，请重新启动监测', updated_at=now())
                write_json(self.data / 'sessions' / session / 'status.json', state)
            state['sources'] = self.sources()
            return state

    def start(self, request):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise ValueError('已有监测任务运行，请先停止')
            item, path = self.source(request.get('source_id'))
            interval = request.get('interval_seconds', 2)
            if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or not 1 <= interval <= 30:
                raise ValueError('检测间隔需在 1 至 30 秒之间')
            options = {}
            for key, default in [('loop', True), ('detect_people', True), ('person_alert', True)]:
                value = request.get(key, default)
                if not isinstance(value, bool):
                    raise ValueError('监测开关格式无效')
                options[key] = value
            config = read_json(self.root / 'config.json')
            if not config.get('fire_smoke', {}).get('enabled'):
                raise ValueError('请先启用本地烟火检测模型')
            person_backend = config.get('person_detection', {}).get('backend')
            if options['detect_people'] and person_backend not in {'damoyolo', 'cix_yolov8s'}:
                raise ValueError('视频监测支持 DAMO-YOLO CPU 或 CIX YOLOv8s NPU 人员检测')
            if options['detect_people'] and person_backend == 'cix_yolov8s' and not Path('/dev/aipu').exists():
                raise ValueError('CIX NPU 设备 /dev/aipu 不可用，请在板卡本机检查设备与驱动')
            session = uuid.uuid4().hex
            folder = self.data / 'sessions' / session
            folder.mkdir(parents=True)
            settings = dict(options, interval_seconds=float(interval), source=item, source_path=str(path), config=config)
            write_json(folder / 'settings.json', settings)
            initial = {'status': 'starting', 'session_id': session, 'source': {k: v for k, v in item.items() if k != 'filename'},
                       'interval_seconds': interval, **options, 'started_at': now(), 'updated_at': now(),
                       'frames_processed': 0, 'events': [], 'message': '正在加载本地检测模型…'}
            write_json(folder / 'status.json', initial)
            if self.log:
                self.log.close()
            self.log = (folder / 'worker.log').open('ab')
            try:
                import os
                self.process = subprocess.Popen([
                    config['fire_smoke']['python'], '-u', str(self.root / 'tools' / 'monitor_video.py'),
                    '--session-dir', str(folder), '--parent-pid', str(os.getpid())
                ], cwd=self.root, stdout=self.log, stderr=self.log, start_new_session=True)
            except OSError:
                self.log.close()
                self.log = None
                raise
            self.session_id = session
            write_json(self.data / 'current.json', {'session_id': session})
            return initial

    def stop(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    import os
                    import signal
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=3)
                session = self._session()
                path = self.data / 'sessions' / session / 'status.json'
                state = read_json(path)
                if state.get('status') in ACTIVE_STATES:
                    state.update(status='stopped', message='监测已停止', updated_at=now())
                    write_json(path, state)
            if self.log:
                self.log.close()
                self.log = None
            return self.status()

    def asset(self, session, kind, filename):
        if not re.fullmatch(r'[a-f0-9]{32}', session):
            raise ValueError('监测编号无效')
        patterns = {'frames': r'frame-\d+\.jpg', 'events': r'[a-f0-9]{32}\.jpg'}
        if kind not in patterns or not re.fullmatch(patterns[kind], filename):
            raise ValueError('监测文件地址无效')
        path = (self.data / 'sessions' / session / kind / filename).resolve()
        if not path.is_relative_to(self.data.resolve()):
            raise ValueError('监测文件地址无效')
        return path
