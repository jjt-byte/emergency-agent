"""Check the offline UI API and worker without opening a listening socket."""
import base64
import io
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from urllib.parse import quote
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent
import webapp


class HTTPApiTests(unittest.TestCase):
    def request(self, method, path, body=None, origin=None,
                host='127.0.0.1:8080', local_address='127.0.0.1'):
        data = json.dumps(body).encode() if body is not None else b''
        class Server:
            server_port = 8080
        class Handler(webapp.Handler):
            def setup(self):
                self.rfile = io.BytesIO(raw)
                self.wfile = output
                self.connection = SimpleNamespace(getsockname=lambda: (local_address, 8080))
            def finish(self):
                pass
            def log_message(self, *args):
                pass
        raw = (f'{method} {path} HTTP/1.0\r\nHost: {host}\r\n'
               f'Content-Length: {len(data)}\r\nContent-Type: application/json\r\n'
               + (f'Origin: {origin}\r\n' if origin else '') + '\r\n').encode() + data
        output = io.BytesIO()
        Handler(None, ('127.0.0.1', 1), Server())
        headers, payload = output.getvalue().split(b'\r\n\r\n', 1)
        code = int(headers.split(b' ', 2)[1])
        return code, payload

    def test_page_and_assets_are_local(self):
        for path in ['/', '/app.js', '/style.css', '/desktop.js', '/desktop.css',
                     '/docs/决赛评分标准对照与网页优化.md', '/docs/技术白皮书与PPT素材.md',
                     '/docs/5分钟Demo演讲稿.md',
                     '/desktop.css']:
            status, body = self.request('GET', quote(path, safe='/'))
            self.assertEqual(status, 200)
            self.assertTrue(body)
        page = self.request('GET', '/')[1].decode()
        resources = []
        class Assets(HTMLParser):
            def handle_starttag(self, tag, attributes):
                for name, value in attributes:
                    if name in {'src', 'href'}:
                        resources.append(value)
        Assets().feed(page)
        self.assertTrue(resources)
        self.assertTrue(all(url.startswith(('/', '#')) for url in resources), resources)

    def test_evaluation_snapshot_reports_measured_scope_and_handles_missing_artifacts(self):
        config = json.loads((webapp.ROOT/'config.json').read_text())
        fixture_root = webapp.ROOT/'benchmarks/historical'
        fixtures = {
            Path(config['person_detection']['validation_file']): json.loads((fixture_root/'person_detection_validation.json').read_text()),
            Path(config['fire_smoke']['validation_file']): json.loads((fixture_root/'fire_smoke_validation.json').read_text()),
            Path(config['model_path']).parent/'benchmark.json': json.loads((fixture_root/'llama-benchmark.json').read_text()),
            webapp.ROOT/'data/monitor/photo-suite-smoke-test.json': json.loads((fixture_root/'photo-suite-smoke-test.json').read_text())}
        for row in fixtures[Path(config['model_path']).parent/'benchmark.json']:
            row['model_filename'] = config['model_path']
        def fixture_read(path):
            if Path(path) in fixtures: return fixtures[Path(path)]
            raise FileNotFoundError(path)
        with patch.object(webapp, 'read_json', side_effect=fixture_read):
            snapshot = webapp.evaluation_snapshot(config)
        self.assertTrue(snapshot['platform']['name'])
        self.assertEqual(snapshot['platform']['npu_available'], Path('/dev/aipu').exists())
        self.assertEqual(snapshot['models']['person_device'], 'CPU · PyTorch')
        self.assertEqual(snapshot['models']['llm_device'], 'CPU · llama.cpp')
        self.assertEqual(snapshot['measurements']['llm_prompt_samples'], 2)
        self.assertEqual(snapshot['measurements']['llm_generation_samples'], 2)
        self.assertEqual(snapshot['measurements']['fire_positive_detected'], 1)
        self.assertEqual(snapshot['measurements']['fire_positive_count'], 3)
        self.assertEqual(snapshot['measurements']['monitor_frames'], 36)
        self.assertEqual(snapshot['sources']['person'], 'data/person_detection_validation.json')
        self.assertNotIn('/home/', json.dumps(snapshot['sources']))
        if snapshot['measurements']['latest_agent_elapsed_seconds'] is not None:
            self.assertGreater(snapshot['measurements']['latest_agent_elapsed_seconds'], 0)
            self.assertIn(snapshot['measurements']['latest_agent_status'], {'completed', 'fallback'})
            self.assertGreater(snapshot['measurements']['latest_agent_rounds'], 0)
        self.assertIn('不计算准确率', snapshot['scope'][1])

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp)/'missing.json'
            sparse = {'person_detection': {'validation_file': str(missing)},
                      'fire_smoke': {'validation_file': str(missing)},
                      'model_path': str(Path(tmp)/'missing.gguf'),
                      'agent': {'enabled': True, 'max_rounds': 99}}
            snapshot = webapp.evaluation_snapshot(sparse)
            self.assertEqual(snapshot['agent']['max_rounds'], 3)
            self.assertEqual(snapshot['measurements']['person_sample_count'], 0)
            self.assertEqual(snapshot['measurements']['fire_positive_count'], 0)
            self.assertIsNone(snapshot['measurements']['llm_generation_tokens_per_second'])

    def test_evaluation_snapshot_is_available_at_local_endpoint(self):
        with patch.object(webapp, 'read_json', return_value={'agent': {'enabled': True}}):
            status, payload = self.request('GET', '/api/evaluation')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)['agent']['enabled'], True)

    def test_reject_fake_upload_and_cross_origin_post(self):
        status, body = self.request('POST', '/api/run', {'task': '检查图片', 'image_base64': base64.b64encode(b'<html>').decode()})
        self.assertEqual(status, 400)
        self.assertIn('JPG', json.loads(body)['error'])
        status, _ = self.request('POST', '/api/run', {}, origin='https://unrelated.example')
        self.assertEqual(status, 403)

    def test_lan_page_can_submit_on_same_origin_and_rejects_other_origins(self):
        local = '192.168.1.25'
        host = local + ':8080'
        with patch.object(webapp, 'submit', return_value={'id': 'a'*32}) as submit:
            status, _ = self.request('POST', '/api/run', {'task': '检查烟火'},
                                     host=host, local_address=local, origin='http://' + host)
            self.assertEqual(status, 202)
            submit.assert_called_once()
        for bad_host, origin in [(host, 'http://192.168.1.26:8080'),
                                 ('unrelated.example:8080', 'http://unrelated.example:8080'),
                                 (local + ':8081', 'http://' + local + ':8081')]:
            with self.subTest(host=bad_host, origin=origin), patch.object(webapp, 'submit') as submit:
                status, _ = self.request('POST', '/api/run', {'task': '检查烟火'},
                                         host=bad_host, local_address=local, origin=origin)
                self.assertEqual(status, 403)
                submit.assert_not_called()

    def test_job_cancel_api_persists_terminal_state_and_checks_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp)
            job_id = 'f'*32
            webapp.write_json(jobs/(job_id+'.json'), {'id': job_id, 'status': 'queued'})
            with patch.object(webapp, 'JOBS', jobs):
                status, body = self.request('POST', '/api/jobs/'+job_id+'/cancel', {},
                                            origin='https://unrelated.example')
                self.assertEqual(status, 403)
                self.assertEqual(webapp.read_json(jobs/(job_id+'.json'))['status'], 'queued')
                status, body = self.request('POST', '/api/jobs/'+job_id+'/cancel', {})
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)['status'], 'cancelled')
                status, body = self.request('GET', '/api/jobs/'+job_id)
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)['status'], 'cancelled')

    def test_evidence_path_cannot_escape_data_directory(self):
        status, _ = self.request('GET', '/data/evidence/%2e%2e%2f%2e%2e%2fconfig.json')
        self.assertEqual(status, 400)
        status, _ = self.request('GET', '/data/uploads/private.jpg')
        self.assertEqual(status, 400)

    def test_actual_worker_saves_report_then_confirmation_updates_all_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = root/'jobs'
            jobs.mkdir()
            job_id = 'a' * 32
            image = root/'image.jpg'
            image.write_bytes((agent.ROOT/'tests/fixtures/blank.jpg').read_bytes())
            webapp.write_json(jobs/f'{job_id}.json', {'id': job_id, 'status': 'queued'})
            # Actual agent subprocess, no hardware/LLM requested; all outputs are isolated.
            import subprocess
            original_run = webapp.run_job_process
            def isolated_agent(job_id, command, log):
                script = (
                    'import sys; from pathlib import Path; import agent; '
                    'agent.DATA=Path(sys.argv[1]); agent.DB=agent.DATA/"knowledge.sqlite3"; '
                    'agent.REPORTS=agent.DATA/"reports"; agent.EVIDENCE=agent.DATA/"evidence"; '
                    'sys.argv=[sys.argv[0]]+sys.argv[2:]; agent.main()'
                )
                isolated = [command[0], '-c', script, str(root)] + command[3:]
                return original_run(job_id, isolated, log)
            with patch.object(webapp, 'JOBS', jobs), patch.object(webapp, 'run_job_process', isolated_agent):
                webapp.worker(job_id, {'task': '离线测试', 'detect': False, 'llm': False}, image)
            state = webapp.read_json(jobs/f'{job_id}.json')
            self.assertEqual(state['status'], 'done', state)
            with patch.multiple(agent, DATA=root, DB=root/'knowledge.sqlite3',
                                REPORTS=root/'reports', EVIDENCE=root/'evidence'):
                status, payload = self.request('GET', '/api/history')
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(payload)[0]['case_id'], state['case_id'])
                status, payload = self.request('POST', '/api/confirm', {'case_id': state['case_id'], 'text': '测试复核记录 <script>'})
                self.assertEqual(status, 200)
                record = json.loads(payload)
                self.assertEqual(record['human_confirmation'], '测试复核记录 <script>')
                self.assertEqual(record['confirmation_history'][0]['text'], '测试复核记录 <script>')
                report = (root/'reports'/f"{state['case_id']}.html").read_text()
                self.assertIn('&lt;script&gt;', report)
                self.assertNotIn('<script>', report)
                self.assertTrue((root/'reports'/f"{state['case_id']}.md").is_file())
            self.assertFalse(image.exists())

    def test_task_acceptance_preserves_image_and_queues_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = (agent.ROOT/'tests/fixtures/blank.jpg').read_bytes()
            with patch.object(webapp, 'JOBS', root/'jobs'), patch.object(webapp, 'UPLOADS', root/'uploads'), patch.object(webapp.POOL, 'submit') as queued:
                status, payload = self.request('POST', '/api/run', {'task': '图片巡查', 'image_name': '../../bad.jpg', 'image_base64': base64.b64encode(raw).decode(), 'llm': False})
                self.assertEqual(status, 202)
                job_id = json.loads(payload)['id']
                self.assertEqual((root/'uploads'/f'{job_id}.jpg').read_bytes(), raw)
                self.assertEqual(webapp.read_json(root/'jobs'/f'{job_id}.json')['status'], 'queued')
                queued.assert_called_once()

    def test_text_only_question_is_accepted_and_worker_omits_image_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(webapp, 'JOBS', root/'jobs'), \
                 patch.object(webapp, 'UPLOADS', root/'uploads'), \
                 patch.object(webapp, 'CONVERSATIONS', root/'conversations'), \
                 patch.object(webapp.POOL, 'submit') as queued:
                status, payload = self.request('POST', '/api/run', {'task': '写两句鼓励救援队员的话', 'llm': True})
                self.assertEqual(status, 202)
                job_id = json.loads(payload)['id']
                self.assertIsNone(queued.call_args.args[3])
                def complete(job_id, command, log):
                    self.assertNotIn('--image', command)
                    self.assertNotIn('--npu-detect', command)
                    self.assertIn('--llm', command)
                    result = Path(command[command.index('--result-file') + 1])
                    result.write_text(json.dumps({'case_id': '20261030-000000-aaaaaa'}))
                    import subprocess
                    return subprocess.CompletedProcess(command, 0)
                with patch.object(webapp, 'run_job_process', side_effect=complete):
                    webapp.worker(job_id, queued.call_args.args[2], None)
                state = webapp.read_json(root/'jobs'/f'{job_id}.json')
                self.assertEqual(state['status'], 'done', state)
                self.assertFalse((root/'uploads').exists())


if __name__ == '__main__':
    unittest.main()
