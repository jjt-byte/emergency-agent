"""Asset import integrity, atomic replacement and archive path boundaries."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import assets


class AssetImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(assets, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        (self.root/'benchmarks').mkdir()
        self.payload = b'fixture-model-data-not-an-actual-model'
        self.name = 'models/test.bin'
        self.manifest(self.name)

    def manifest(self, name):
        (self.root/'benchmarks/models_manifest.json').write_text(json.dumps({'files':[
            {'path': name, 'bytes':len(self.payload), 'sha256':hashlib.sha256(self.payload).hexdigest()}]}))

    def archive(self, name=None, payload=None, link=False):
        path=self.root/'assets.tar'
        with tarfile.open(path,'w') as out:
            info=tarfile.TarInfo(name or self.name)
            if link:
                info.type=tarfile.SYMTYPE;info.linkname='/tmp/untrusted'
                out.addfile(info)
            else:
                data=self.payload if payload is None else payload
                info.size=len(data);out.addfile(info,io.BytesIO(data))
        return path

    def test_valid_archive_import_and_verify(self):
        assets.import_archive(self.archive(),'models')
        assets.verify('models')
        self.assertEqual((self.root/self.name).read_bytes(),self.payload)

    def test_bad_hash_preserves_existing_file_and_removes_partial(self):
        target=self.root/self.name;target.parent.mkdir();target.write_bytes(b'old')
        with self.assertRaisesRegex(ValueError,'SHA256'):
            assets.import_archive(self.archive(payload=b'x'*len(self.payload)),'models')
        self.assertEqual(target.read_bytes(),b'old')
        self.assertFalse(list(self.root.rglob('*.part')))

    def test_links_and_traversal_are_rejected(self):
        with self.assertRaises(ValueError):assets.import_archive(self.archive(link=True),'models')
        name='../outside.bin';self.manifest(name)
        with self.assertRaises(ValueError):assets.import_archive(self.archive(name=name),'models')
        self.assertFalse((self.root.parent/'outside.bin').exists())

    def test_missing_declared_file_is_rejected(self):
        path=self.root/'empty.tar'
        with tarfile.open(path,'w'):pass
        with self.assertRaisesRegex(ValueError,'缺少资源'):assets.import_archive(path,'models')


if __name__=='__main__':unittest.main()
