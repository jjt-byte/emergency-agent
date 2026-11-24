"""Regression checks for clean deployments and external runtime reuse."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import configure, run


class RuntimeSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('AGENT_PYTHON', None)

    def config(self, **values):
        (self.root / 'config.json').write_text(json.dumps(values), encoding='utf-8')

    def test_configured_external_runtime_has_priority_over_local_venv(self):
        local = self.root / '.venv/bin/python'
        local.parent.mkdir(parents=True)
        local.touch()
        external = self.root / 'external environment/python'
        self.config(runtime_python=str(external), vision_python=str(local))
        self.assertEqual(run.runtime_python(self.root), str(external))

    def test_previous_configuration_uses_vision_python(self):
        self.config(vision_python='.venv-fire/bin/python')
        self.assertEqual(run.runtime_python(self.root), str(self.root / '.venv-fire/bin/python'))

    def test_environment_override_accepts_executable_command(self):
        self.config(runtime_python='/unused/python')
        with patch.dict(os.environ, AGENT_PYTHON='python3'), patch.object(
                run.shutil, 'which', return_value=sys.executable):
            self.assertEqual(run.runtime_python(self.root), sys.executable)

    def test_relative_runtime_resolves_against_its_own_checkout(self):
        self.config(runtime_python='.venv/bin/python')
        with patch.object(run.shutil, 'which', return_value='/different/checkout/.venv/bin/python'):
            self.assertEqual(run.runtime_python(self.root), str(self.root / '.venv/bin/python'))

    def test_unconfigured_checkout_uses_local_venv_then_current_python(self):
        self.assertEqual(run.runtime_python(self.root), sys.executable)
        local = self.root / '.venv/bin/python'
        local.parent.mkdir(parents=True)
        local.touch()
        self.assertEqual(run.runtime_python(self.root), str(local))

    def test_invalid_configuration_is_reported(self):
        (self.root / 'config.json').write_text('{broken', encoding='utf-8')
        with self.assertRaises(json.JSONDecodeError):
            run.runtime_python(self.root)

    def test_launcher_preserves_entrypoint_and_argument_boundaries(self):
        python = str(self.root / 'external environment/python')
        arguments = ['run.py', 'webapp.py', '--host', '127.0.0.1', '--port', '8082']
        with patch.object(sys, 'argv', arguments), patch.object(run, 'runtime_python',
                return_value=python), patch.object(run.os, 'execv') as launch:
            run.main()
        launch.assert_called_once_with(python, [python, str(run.ROOT / 'webapp.py'),
                                              '--host', '127.0.0.1', '--port', '8082'])

    def test_configure_records_the_web_and_detector_runtime(self):
        (self.root / 'config.example.json').write_bytes(
            (configure.ROOT / 'config.example.json').read_bytes())
        with patch.object(configure, 'ROOT', self.root):
            config = configure.configure(sys.executable)
        self.assertEqual(config['runtime_python'], sys.executable)
        self.assertEqual(config['person_detection']['python'], config['runtime_python'])
        self.assertEqual(config['fire_smoke']['python'], config['runtime_python'])


if __name__ == '__main__':
    unittest.main()
