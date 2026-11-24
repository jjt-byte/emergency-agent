"""Downloads are verified as files, never executed as model code."""
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import download_fire_smoke as download


class DownloadTests(unittest.TestCase):
    def candidate(self, payload):
        return {'repo': 'test/fire-smoke', 'file': 'weights/model.onnx', 'revision': 'abc123',
                'license': 'apache-2.0', 'size_bytes': len(payload),
                'sha256_expected': hashlib.sha256(payload).hexdigest(),
                'model_page': 'https://huggingface.co/test/fire-smoke',
                'download_url': 'https://huggingface.co/test/fire-smoke/resolve/abc123/weights/model.onnx'}

    def test_hash_and_provenance_written_after_complete_download(self):
        raw = b'example-model-bytes-used-only-as-test-fixture'
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(download, 'urlopen', return_value=io.BytesIO(raw)), patch.object(download, 'fetch', side_effect=OSError('no card')):
                with contextlib.redirect_stdout(io.StringIO()):
                    manifest = download.download(self.candidate(raw), Path(tmp))
            self.assertEqual(Path(manifest['local_path']).read_bytes(), raw)
            self.assertEqual(manifest['actual_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(manifest['status'], 'downloaded_unvalidated')
            self.assertIn('resolve/abc123/', (Path(manifest['local_path']).parent/'DOWNLOAD_LINKS.md').read_text())

    def test_corrupt_download_never_promoted_to_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self.candidate(b'expected')
            with patch.object(download, 'urlopen', return_value=io.BytesIO(b'corrupt!')):
                with self.assertRaises(ValueError):
                    download.download(candidate, Path(tmp))
            self.assertFalse(list(Path(tmp).rglob('*.onnx')))
            self.assertFalse(list(Path(tmp).rglob('*.part')))
            self.assertFalse(list(Path(tmp).rglob('source.json')))


if __name__ == '__main__':
    unittest.main()
