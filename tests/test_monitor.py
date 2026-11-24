"""Check automatic alerts, local video boundaries, APIs and worker faults."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor
import webapp
from tools import monitor_video


class AlertTests(unittest.TestCase):
    fire = [{'label': 'fire', 'confidence': .8, 'box': [1, 2, 30, 40]}]

    def test_first_hit_notifies_and_repeated_hits_escalate_same_event(self):
        tracker = monitor.AlarmTracker()
        first = tracker.update(self.fire, 0, 'first')[0]
        self.assertEqual(first['stage'], 'first_seen')
        self.assertEqual(first['level'], 'warning')
        second = tracker.update(self.fire, 2, 'second')[0]
        self.assertEqual(second['id'], first['id'])
        self.assertEqual(second['stage'], 'persistent')
        self.assertEqual(second['level'], 'critical')
        for position in (4, 6, 8):
            event = tracker.update(self.fire, position, 'later')[0]
            self.assertEqual(event['id'], first['id'])
        self.assertEqual(event['samples'], 5)

    def test_three_clear_samples_end_event_and_allow_a_new_one(self):
        tracker = monitor.AlarmTracker()
        first = tracker.update(self.fire, 0, 'first')[0]
        self.assertEqual(tracker.update([], 2, 'second'), [])
        self.assertEqual(tracker.update([], 4, 'third'), [])
        ended = tracker.update([], 6, 'fourth')[0]
        self.assertEqual(ended['status'], 'ended')
        self.assertIn('不表示现场安全', ended['end_reason'])
        fresh = tracker.update(self.fire, 8, 'fifth')[0]
        self.assertNotEqual(fresh['id'], first['id'])
        self.assertEqual(fresh['stage'], 'first_seen')

    def test_people_are_optional_and_shutdown_does_not_claim_safety(self):
        people = [{'label': 'person', 'confidence': .9}]
        self.assertEqual(monitor.AlarmTracker(False).update(people, 0, 'now'), [])
        tracker = monitor.AlarmTracker(True)
        event = tracker.update(people, 0, 'now')[0]
        self.assertEqual(event['level'], 'notice')
        ended = tracker.end('later', '监测结束，后续画面未检测')[0]
        self.assertEqual(ended['end_reason'], '监测结束，后续画面未检测')
        self.assertEqual(tracker.end('later', 'end'), [])

    def test_separated_hits_do_not_count_as_consecutive(self):
        tracker = monitor.AlarmTracker()
        tracker.update(self.fire, 0, 'now')
        tracker.update([], 2, 'now')
        self.assertEqual(tracker.update(self.fire, 4, 'now')[0]['stage'], 'first_seen')


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.controller = monitor.Controller(self.root)
        self.video = self.root/'data/monitor/videos'
        self.video.mkdir(parents=True)
        (self.video/'demo.mp4').write_bytes(b'fixture')
        monitor.write_json(self.video/'demo.json', {'id':'demo','filename':'demo.mp4','name':'演示',
                                                  'duration_seconds':32,'width':960,'height':640})
        monitor.write_json(self.root/'config.json', {'fire_smoke': {'enabled':True, 'python':sys.executable},
                                                     'person_detection':{'backend':'damoyolo'}})

    def test_paths_and_options_rejected_before_starting_process(self):
        for source in ('../config', '/etc/passwd', None, 'demo/../x'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.controller.source(source)
        for interval in (True, 0, 31, float('nan'), float('inf'), '2'):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.controller.start({'source_id':'demo', 'interval_seconds':interval})
        with self.assertRaises(ValueError):
            self.controller.start({'source_id':'demo','loop':'yes'})
        with self.assertRaises(ValueError):
            self.controller.asset('a'*32, 'frames', '../../config.json')
        with self.assertRaises(ValueError):
            self.controller.asset('a'*32, 'events', 'e'*32+'.json')

    def test_import_streams_bytes_and_cleans_up_failed_upload(self):
        with patch.object(monitor,'probe_video',return_value={'duration_seconds':2,'width':640,'height':480}):
            result=self.controller.upload(io.BytesIO(b'video'),5,'../../my-video.mp4')
            self.assertEqual(result['name'],'my-video.mp4')
            metadata,path=self.controller.source(result['id'])
            self.assertEqual(path.read_bytes(),b'video')
            self.assertNotIn('filename',result)
        before=set(self.video.iterdir())
        with self.assertRaisesRegex(ValueError,'中断'):
            self.controller.upload(io.BytesIO(b'partial'),100,'broken.mp4')
        self.assertEqual(set(self.video.iterdir()),before)
        with self.assertRaises(ValueError):
            self.controller.upload(io.BytesIO(b''),monitor.MAX_VIDEO_BYTES+1,'big.mp4')
        with patch.object(monitor,'probe_video',side_effect=ValueError('invalid')):
            with self.assertRaises(ValueError):
                self.controller.upload(io.BytesIO(b'no-video'),8,'fake.mp4')
        self.assertEqual(set(self.video.iterdir()),before)

    def test_start_stop_and_abandoned_session(self):
        process=Mock()
        process.poll.return_value=None
        with patch.object(monitor.subprocess,'Popen',return_value=process):
            state=self.controller.start({'source_id':'demo','loop':False})
        self.assertEqual(state['status'],'starting')
        self.assertEqual(self.controller.status()['status'],'starting')
        with self.assertRaisesRegex(ValueError,'已有监测'):
            self.controller.start({'source_id':'demo'})
        process.wait.side_effect=lambda **_: setattr(process.poll,'return_value',0)
        stopped=self.controller.stop()
        self.assertEqual(stopped['status'],'stopped')
        process.terminate.assert_called_once()
        path=self.controller.data/'sessions'/state['session_id']/'status.json'
        monitor.write_json(path,dict(state,status='running'))
        fresh=monitor.Controller(self.root)
        self.assertEqual(fresh.status()['status'],'failed')
        self.assertIn('停止',fresh.status()['error'])

    def test_metadata_cannot_escape_video_directory(self):
        monitor.write_json(self.video/'demo.json', {'filename':'../../../config.json'})
        with self.assertRaises(ValueError):
            self.controller.source('demo')

    def test_video_accepts_cix_person_backend_and_checks_device(self):
        config = monitor.read_json(self.root/'config.json')
        config['person_detection'] = {'backend':'cix_yolov8s', 'python':'/noe/python',
                                      'model_path':'/models/yolov8s.cix'}
        monitor.write_json(self.root/'config.json', config)
        original_exists = Path.exists
        def without_npu(path):
            return False if str(path) == '/dev/aipu' else original_exists(path)
        with patch.object(Path, 'exists', without_npu), patch.object(monitor.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError, '/dev/aipu'):
                self.controller.start({'source_id':'demo'})
            launch.assert_not_called()
        def with_npu(path):
            return True if str(path) == '/dev/aipu' else original_exists(path)
        process = Mock()
        process.poll.return_value = None
        with patch.object(Path, 'exists', with_npu), patch.object(monitor.subprocess,'Popen',return_value=process):
            state = self.controller.start({'source_id':'demo'})
        saved = monitor.read_json(self.controller.data/'sessions'/state['session_id']/'settings.json')
        self.assertEqual(saved['config']['person_detection']['backend'], 'cix_yolov8s')
        self.assertEqual(saved['config']['person_detection']['python'], '/noe/python')
        process.wait.side_effect=lambda **_: setattr(process.poll,'return_value',0)
        self.controller.stop()

    def test_probe_timeout_becomes_a_readable_validation_error(self):
        with patch.object(monitor.subprocess,'run',side_effect=subprocess.TimeoutExpired('probe',15)):
            with self.assertRaisesRegex(ValueError,'检查超时'):
                monitor.probe_video('video.mp4')

    def test_worker_failure_is_not_reported_as_no_detections(self):
        session=self.controller.data/'sessions'/('a'*32)
        session.mkdir(parents=True)
        settings={'config':{'fire_smoke':{},'cpu_affinity':''},'detect_people':False,'person_alert':False,
                  'source':{'duration_seconds':10},'loop':False,'interval_seconds':2,'source_path':'video.mp4'}
        monitor.write_json(session/'settings.json',settings)
        monitor.write_json(session/'status.json',{'status':'starting','frames_processed':0,'events':[]})
        detector=Mock()
        detector.receive.side_effect=[{'ready':True},RuntimeError('模型推理失败')]
        def capture(_video,_position,output):
            output.write_bytes(b'frame')
        with patch.object(monitor_video,'parent_lifetime'), patch.object(monitor_video,'Detector',return_value=detector), \
                patch.object(monitor_video,'capture',side_effect=capture), patch.object(monitor_video.signal,'signal'):
            monitor_video.run(session,123)
        state=monitor.read_json(session/'status.json')
        self.assertEqual(state['status'],'failed')
        self.assertIn('模型推理失败',state['error'])
        self.assertNotIn('people_count',state)
        self.assertEqual(state['events'],[])
        detector.close.assert_called_once()


class DetectorRoutingTests(unittest.TestCase):
    def test_npu_uses_noe_interpreter_with_cix_loader(self):
        settings={'backend':'cix_yolov8s','python':'/noe/python',
                  'model_path':'/models/yolov8s.cix','threshold':.4,'nms_threshold':.45}
        with patch('tools.detect_yolov8_npu.LocalDetector') as load:
            monitor_video.load_detector('person',settings)
            load.assert_called_once_with('/models/yolov8s.cix',.4,.45)
        process=Mock()
        process.poll.return_value=0
        with patch.object(monitor_video.subprocess,'Popen',return_value=process) as launch, \
                patch.object(monitor_video.selectors,'DefaultSelector'):
            detector=monitor_video.Detector('person',settings,threading.Event())
            command=launch.call_args.args[0]
            self.assertEqual(command[0],'/noe/python')
            self.assertEqual(json.loads(command[command.index('--detector-settings')+1]),settings)
            detector.close()
            process.stdin.close.assert_called_once()
            process.stdout.close.assert_called_once()

    def test_cpu_loader_preserves_person_threshold_and_fire_settings(self):
        with patch('tools.detect_person.LocalDetector') as load:
            monitor_video.load_detector('person',{'backend':'damoyolo','model_dir':'/person'})
            load.assert_called_once_with('/person',.5,640,8)
        with patch('tools.detect_fire_smoke.LocalDetector') as load:
            monitor_video.load_detector('fire_smoke',{'model_dir':'/fire','threshold':.6,'threads':4})
            load.assert_called_once_with('/fire',.6,640,4)
        with self.assertRaises(ValueError):
            monitor_video.load_detector('person',{'backend':'unsupported'})

    def test_clean_exit_sends_sentinel_before_terminating_npu_worker(self):
        process=Mock()
        process.poll.return_value=None
        with patch.object(monitor_video.subprocess,'Popen',return_value=process), \
                patch.object(monitor_video.selectors,'DefaultSelector'):
            detector=monitor_video.Detector('person',{'python':'/noe/python'},threading.Event())
            detector.close()
        process.stdin.write.assert_called_once_with('null\n')
        process.terminate.assert_not_called()

    def test_worker_cleans_npu_engine_after_end_of_input(self):
        detector=Mock()
        with patch.object(monitor_video,'parent_lifetime'), \
                patch.object(monitor_video,'load_detector',return_value=detector), \
                patch.object(sys,'stdin',io.StringIO('null\n')), \
                patch.object(sys,'stdout',io.StringIO()) as output:
            monitor_video.detector_process('person',{},123)
            self.assertEqual(json.loads(output.getvalue()),{'ready':True})
        detector.close.assert_called_once()


class MonitorApiTests(unittest.TestCase):
    def request(self,method,path,body=b'',extra='',origin=None):
        output=io.BytesIO()
        raw=(f'{method} {path} HTTP/1.0\r\nHost: 127.0.0.1:8080\r\nContent-Length: {len(body)}\r\n'
             + (f'Origin: {origin}\r\n' if origin else '') + extra + '\r\n').encode()+body
        class Handler(webapp.Handler):
            def setup(self):
                self.rfile=io.BytesIO(raw); self.wfile=output
                self.connection=SimpleNamespace(getsockname=lambda:('127.0.0.1',8080))
            def finish(self): pass
            def log_message(self,*args): pass
        Handler(None,('127.0.0.1',1),SimpleNamespace(server_port=8080))
        headers,payload=output.getvalue().split(b'\r\n\r\n',1)
        return int(headers.split()[1]),payload

    def test_status_static_script_and_post_routes(self):
        fake=Mock()
        fake.status.return_value={'status':'idle','sources':[]}
        fake.start.return_value={'status':'starting'}
        fake.stop.return_value={'status':'stopped'}
        with patch.object(webapp,'MONITOR',fake):
            self.assertEqual(self.request('GET','/api/monitor')[0],200)
            self.assertEqual(self.request('POST','/api/monitor/start',b'{"source_id":"demo"}')[0],202)
            fake.start.assert_called_once_with({'source_id':'demo'})
            self.assertEqual(self.request('POST','/api/monitor/stop',b'{}')[0],200)
            self.assertEqual(self.request('POST','/api/monitor/stop',b'{}',origin='http://unrelated.example')[0],403)
            fake.stop.assert_called_once()
        self.assertEqual(self.request('GET','/monitor.js')[0],200)

    def test_binary_video_upload_uses_video_limit_and_local_origin(self):
        fake=Mock()
        fake.upload.return_value={'id':'a'*32}
        with patch.object(webapp,'MONITOR',fake):
            code,_=self.request('POST','/api/monitor/video',b'video','X-File-Name: test.mp4\r\n')
            self.assertEqual(code,201)
            args=fake.upload.call_args.args
            self.assertEqual(args[1:],(5,'test.mp4'))
            self.assertEqual(self.request('POST','/api/monitor/video',b'video',origin='http://unrelated.example')[0],403)
            self.assertEqual(fake.upload.call_count,1)


if __name__=='__main__':
    unittest.main()
