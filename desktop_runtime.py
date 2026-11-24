"""Start and stop only the loopback backend owned by the desktop application."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from urllib.request import ProxyHandler, build_opener

from scripts.run import runtime_python

ROOT = Path(__file__).resolve().parent
APP_ID = 'cn.yingjixianxing.Desktop'
OPENER = build_opener(ProxyHandler({}))


def local_json(url, timeout=2):
    with OPENER.open(url, timeout=timeout) as response:
        return json.load(response)


class Backend:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.url = None
        self.process = None
        self.owned = False
        self.temporary = None
        self.log = None
        self.lock = threading.RLock()

    def start(self, reuse=True, timeout=20):
        with self.lock:
            if reuse:
                url = 'http://127.0.0.1:8080'
                try:
                    info = local_json(url + '/api/app-info', timeout=.5)
                    if (info.get('application') == APP_ID and info.get('protocol') == 1
                            and info.get('project_root') == str(self.root.resolve())):
                        self.url = url
                        return url
                except (OSError, ValueError):
                    pass
            directory = self.root / 'data/desktop'
            directory.mkdir(parents=True, exist_ok=True)
            self.temporary = tempfile.TemporaryDirectory(prefix='launch-', dir=directory)
            ready = Path(self.temporary.name) / 'ready.json'
            self.log = (directory / 'backend.log').open('ab')
            try:
                backend_python = runtime_python(self.root)
                self.process = subprocess.Popen([
                    backend_python, '-u', str(self.root / 'webapp.py'),
                    '--host', '127.0.0.1', '--port', '0', '--ready-file', str(ready),
                ], cwd=self.root, stdout=self.log, stderr=self.log, process_group=0)
                self.owned = True
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError('本地服务启动失败，请查看 data/desktop/backend.log')
                    if ready.is_file():
                        info = json.loads(ready.read_text())
                        if (info.get('pid') != self.process.pid or info.get('host') != '127.0.0.1'
                                or not isinstance(info.get('port'), int)):
                            raise RuntimeError('本地服务启动信息无效')
                        self.url = 'http://127.0.0.1:' + str(info['port'])
                        identity = local_json(self.url + '/api/app-info')
                        if (identity.get('application') != APP_ID or identity.get('protocol') != 1
                                or identity.get('project_root') != str(self.root.resolve())):
                            raise RuntimeError('本地服务身份检查失败')
                        return self.url
                    time.sleep(.1)
                raise RuntimeError('本地服务启动超时，请查看 data/desktop/backend.log')
            except Exception:
                self.stop()
                raise

    def stop(self):
        with self.lock:
            process = self.process
            if self.owned and process is not None:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        pass
                # This private process group contains only our owned service and workers.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if process.poll() is None:
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=3)
            self.process = None
            self.owned = False
            if self.log:
                self.log.close()
                self.log = None
            if self.temporary:
                self.temporary.cleanup()
                self.temporary = None

    def has_active_jobs(self):
        paths = list((self.root / 'data/web_jobs').glob('*.json'))
        paths += list((self.root / 'data/batches').glob('*/status.json'))
        for path in paths:
            try:
                if json.loads(path.read_text()).get('status') in {'queued', 'preparing', 'running'}:
                    return True
            except (OSError, ValueError):
                continue
        return False
