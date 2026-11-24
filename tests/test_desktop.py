"""Desktop backend ownership, readiness and failure cleanup without sockets."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import desktop_runtime as runtime


class DesktopBackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backend = runtime.Backend(self.root)
        self.identity = {'application': runtime.APP_ID, 'protocol': 1,
                         'project_root': str(self.root.resolve())}

    def test_recognized_external_service_is_reused_and_never_stopped(self):
        with patch.object(runtime, 'local_json', return_value=self.identity), patch.object(
                runtime.subprocess, 'Popen') as launch, patch.object(runtime.os, 'killpg') as kill:
            self.assertEqual(self.backend.start(), 'http://127.0.0.1:8080')
            self.assertFalse(self.backend.owned)
            self.backend.stop()
        launch.assert_not_called()
        kill.assert_not_called()

    def test_other_checkout_is_not_reused_and_start_failure_cleans_up(self):
        other = dict(self.identity, project_root='/some/other/project')
        with patch.object(runtime, 'local_json', return_value=other), patch.object(
                runtime.subprocess, 'Popen', side_effect=OSError('cannot launch')) as launch:
            with self.assertRaisesRegex(OSError, 'cannot launch'):
                self.backend.start()
        launch.assert_called_once()
        self.assertFalse(self.backend.owned)
        self.assertIsNone(self.backend.log)
        self.assertIsNone(self.backend.temporary)

    def test_owned_service_reads_actual_port_and_stops_its_private_group(self):
        process = Mock(pid=32123)
        process.poll.side_effect = [None, None, 0]
        def launch(arguments, **options):
            ready = Path(arguments[arguments.index('--ready-file') + 1])
            ready.write_text(json.dumps({'host': '127.0.0.1', 'port': 43210, 'pid': process.pid}))
            self.assertEqual(options['process_group'], 0)
            self.assertEqual(arguments[arguments.index('--port') + 1], '0')
            return process
        with patch.object(runtime.subprocess, 'Popen', side_effect=launch), patch.object(
                runtime, 'local_json', return_value=self.identity), patch.object(
                runtime.os, 'killpg') as kill:
            self.assertEqual(self.backend.start(reuse=False), 'http://127.0.0.1:43210')
            self.assertTrue(self.backend.owned)
            self.backend.stop()
        process.send_signal.assert_called_once_with(signal.SIGINT)
        kill.assert_called_once_with(process.pid, signal.SIGTERM)
        self.assertFalse(self.backend.owned)
        self.assertIsNone(self.backend.temporary)

    def test_early_exit_is_reported_and_resources_are_closed(self):
        process = Mock(pid=32124)
        process.poll.return_value = 1
        with patch.object(runtime.subprocess, 'Popen', return_value=process), patch.object(
                runtime.os, 'killpg'):
            with self.assertRaisesRegex(RuntimeError, '启动失败'):
                self.backend.start(reuse=False)
        self.assertIsNone(self.backend.process)
        self.assertIsNone(self.backend.log)
        self.assertIsNone(self.backend.temporary)

    def test_unresponsive_owned_service_escalates_to_kill(self):
        process = Mock(pid=32125)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('backend', 8),
                                   subprocess.TimeoutExpired('backend', 3), 0]
        self.backend.process, self.backend.owned = process, True
        with patch.object(runtime.os, 'killpg') as kill:
            self.backend.stop()
        self.assertEqual(kill.call_args_list[-1].args, (process.pid, signal.SIGKILL))

    def test_exit_detection_covers_analysis_and_batch_tasks(self):
        jobs = self.root / 'data/web_jobs'
        jobs.mkdir(parents=True)
        (jobs / 'bad.json').write_text('broken')
        self.assertFalse(self.backend.has_active_jobs())
        task = jobs / 'task.json'
        task.write_text(json.dumps({'status': 'running'}))
        self.assertTrue(self.backend.has_active_jobs())
        task.write_text(json.dumps({'status': 'done'}))
        batch = self.root / 'data/batches/example/status.json'
        batch.parent.mkdir(parents=True)
        batch.write_text(json.dumps({'status': 'preparing'}))
        self.assertTrue(self.backend.has_active_jobs())
        batch.write_text(json.dumps({'status': 'done'}))
        self.assertFalse(self.backend.has_active_jobs())


if __name__ == '__main__':
    unittest.main()
