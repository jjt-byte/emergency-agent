"""Regression checks for legacy conversations, failed jobs and real cancellation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import agent
import webapp


class JobLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.jobs = self.root/'jobs'
        self.jobs.mkdir()
        self.job_id = 'a'*32
        self.path = self.jobs/(self.job_id+'.json')
        webapp.write_json(self.path, {'id': self.job_id, 'status': 'queued'})

    def test_legacy_report_is_enriched_before_conversation_memory(self):
        case_id = '20261029-161458-003cfb'
        original = json.dumps({'case_id': case_id, 'task': '这张图有多少个人？', 'detections': [{'label': 'person', 'confidence': .9}], 'detection_source': 'cix_ssd_300_npu', 'tool_error': False}).encode()
        record = json.loads(original)
        self.assertNotIn('task_answer', record)
        reports = self.root/'reports'
        reports.mkdir()
        (reports/(case_id+'.json')).write_bytes(original)
        session = {'id': 'b'*32, 'case_ids': [case_id]}
        with patch.object(agent, 'REPORTS', reports):
            context, image = webapp.conversation_context(session)
        self.assertIsNone(image)
        self.assertTrue(context['history'][0]['answer'])
        self.assertEqual((reports/(case_id+'.json')).read_bytes(), original)

    def test_unexpected_worker_exception_marks_failure_and_cleans_upload(self):
        upload = self.root/'upload.jpg'
        upload.write_bytes(b'fixture')
        session = {'id': 'b'*32, 'case_ids': []}
        conversations = self.root/'conversations'
        webapp.write_json(conversations/(session['id']+'.json'), session)
        with (patch.object(webapp, 'JOBS', self.jobs),
              patch.object(webapp, 'CONVERSATIONS', conversations),
              patch.object(webapp, 'conversation_context', side_effect=KeyError('unexpected_field'))):
            webapp.worker(self.job_id, {'task': 'fire', 'agent': True, 'conversation_id': session['id']}, upload)
        state = webapp.read_json(self.path)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('KeyError', state['error'])
        self.assertFalse(upload.exists())
        self.assertIn('unexpected_field', self.path.with_suffix('.log').read_text())

    def test_cancelled_queued_job_never_launches_process(self):
        with patch.object(webapp, 'JOBS', self.jobs), patch.object(webapp, 'run_job_process') as run:
            self.assertEqual(webapp.cancel_job(self.job_id)['status'], 'cancelled')
            webapp.worker(self.job_id, {'task': 'unused'}, None)
        run.assert_not_called()
        self.assertEqual(webapp.read_json(self.path)['status'], 'cancelled')

    def test_finished_job_is_not_overwritten_by_cancel(self):
        state = {'id': self.job_id, 'status': 'done', 'case_id': 'saved-record'}
        webapp.write_json(self.path, state)
        with patch.object(webapp, 'JOBS', self.jobs), patch.object(webapp, 'stop_job_process') as stop:
            self.assertEqual(webapp.cancel_job(self.job_id), state)
        stop.assert_not_called()

    def test_running_cancellation_stops_owned_process_and_nested_child(self):
        ready = self.root/'child.pid'
        original_run = webapp.run_job_process
        script = (
            'import subprocess,sys,time; from pathlib import Path; '
            'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
            'Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)'
        )

        def run(job_id, command, log):
            return original_run(job_id, [sys.executable, '-c', script, str(ready)], log)

        with patch.object(webapp, 'JOBS', self.jobs), patch.object(webapp, 'run_job_process', side_effect=run):
            thread = threading.Thread(target=webapp.worker,
                                      args=(self.job_id, {'task': 'cancel test', 'llm': False}, None))
            thread.start()
            try:
                deadline = time.monotonic()+5
                while not ready.is_file() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(ready.is_file(), webapp.read_json(self.path))
                child_pid = int(ready.read_text())
                process = webapp.RUNNING_JOBS[self.job_id]
                self.assertEqual(os.getpgid(child_pid), process.pid)
                self.assertEqual(webapp.cancel_job(self.job_id)['status'], 'cancelled')
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertIsNotNone(process.poll())
                child_status = Path(f'/proc/{child_pid}/stat')
                if child_status.is_file():
                    self.assertEqual(child_status.read_text().split()[2], 'Z')
                self.assertEqual(webapp.read_json(self.path)['status'], 'cancelled')
                self.assertNotIn(self.job_id, webapp.RUNNING_JOBS)
            finally:
                if thread.is_alive():
                    webapp.cancel_job(self.job_id)
                    thread.join(5)

    def test_real_timeout_stops_process_and_marks_failure(self):
        original_run = webapp.run_job_process

        def run(job_id, command, log):
            return original_run(job_id, [sys.executable, '-c', 'import time; time.sleep(30)'], log)

        with (patch.object(webapp, 'JOBS', self.jobs),
              patch.object(webapp, 'JOB_TIMEOUT', .1),
              patch.object(webapp, 'run_job_process', side_effect=run)):
            webapp.worker(self.job_id, {'task': 'timeout test', 'llm': False}, None)
        state = webapp.read_json(self.path)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('已停止', state['error'])
        self.assertNotIn(self.job_id, webapp.RUNNING_JOBS)


if __name__ == '__main__':
    unittest.main()
