"""Human model routing, coordinate recovery, failure handling and provenance."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import agent
from tools import detect_person


class PersonDetectorTests(unittest.TestCase):
    settings = {'backend': 'damoyolo', 'python': '/local/python',
                'model_dir': '/local/person', 'threshold': .5, 'max_side': 640, 'threads': 8}

    def test_labels_filtering_and_original_coordinates(self):
        raw = {'class_ids': ['person', 'person'], 'scores': [.8, .3],
               'boxes': [[5, 10, 450, 300], [1, 1, 20, 20]]}
        result = detect_person.normalize_result(raw, 1000, 500, (.5, .5))
        self.assertEqual(result, [{'label': 'person', 'confidence': .8, 'box': [10., 20., 900., 500.]}])
        self.assertEqual(detect_person.normalize_result({'class_ids': [], 'scores': [], 'boxes': []}, 100, 100), [])
        for malformed in [dict(raw, class_ids=['face', 'person']), dict(raw, scores=[float('nan'), .3]),
                          dict(raw, boxes=[[0, 1, 2, float('inf')], [1, 1, 20, 20]]),
                          dict(raw, class_ids=['person']), dict(raw, boxes=None)]:
            with self.subTest(output=malformed), self.assertRaises((ValueError, TypeError)):
                detect_person.normalize_result(malformed, 1000, 500)

    def test_damoyolo_dispatch_does_not_require_npu_and_rejects_invalid_outputs(self):
        payload = {'detections': [{'label': 'person', 'confidence': .9, 'box': [1, 2, 30, 40]}],
                   'image_size': [100, 100], 'source': detect_person.SOURCE}
        def completed(value):
            return argparse.Namespace(returncode=0, stdout=json.dumps(value), stderr='')
        with patch.dict(agent.CONFIG, {'person_detection': self.settings}), \
             patch.object(agent.subprocess, 'run', return_value=completed(payload)) as run:
            result, raw = agent.run_npu_detector('photo.jpg')
            self.assertEqual(result[0]['label'], 'person')
            self.assertEqual(raw['image_size'], [100, 100])
            self.assertIn('detect_person.py', run.call_args.args[0][1])
            self.assertEqual(run.call_args.args[0][0], '/local/python')
            for bad in [dict(payload, image_size=[0, 100]),
                        dict(payload, detections=[{'label': 'face', 'confidence': .9, 'box': [1, 2, 3, 4]}]),
                        dict(payload, detections=[{'label': 'person', 'confidence': .9}])]:
                run.return_value = completed(bad)
                with self.assertRaises(ValueError): agent.run_npu_detector('photo.jpg')
            run.return_value = argparse.Namespace(returncode=1, stdout='', stderr='person runtime failed')
            with self.assertRaisesRegex(RuntimeError, 'person runtime failed'):
                agent.run_npu_detector('photo.jpg')

    def test_active_person_model_is_recorded_and_old_evidence_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root/'person';model_dir.mkdir()
            model = model_dir/detect_person.WEIGHT;model.write_bytes(b'fixture')
            settings = dict(self.settings, model_dir=str(model_dir))
            args = argparse.Namespace(task='有几个人', image=str(agent.ROOT/'tests/fixtures/blank.jpg'),
                                      observation='', npu_detect=True, detections=None, tool_error=False,
                                      confirm=None, confirm_after=False, llm=False)
            boxes = [{'label': 'person', 'confidence': .8, 'box': [1, 2, 30, 40]}]
            payload = {'detections': boxes, 'image_size': [640, 427], 'threshold': .5}
            with patch.multiple(agent, DATA=root, DB=root/'db.sqlite3', REPORTS=root/'reports', EVIDENCE=root/'evidence'), \
                 patch.dict(agent.CONFIG, {'person_detection': settings}), \
                 patch.object(agent, 'run_npu_detector', return_value=(boxes, payload)), \
                 patch.object(agent, 'run_fire_smoke_detector') as fire, \
                 contextlib.redirect_stdout(io.StringIO()):
                record = agent.run_case(args)
            fire.assert_not_called()
            self.assertEqual(record['detection_source'], 'damoyolo_person_cpu')
            self.assertEqual(record['vision_model_path'], str(model))
            self.assertEqual(record['vision_model_sha256'], agent.sha256(model))
            self.assertEqual(record['person_statistics']['estimated_count'], 1)
            self.assertEqual(record['person_statistics']['detection_threshold'], .5)
            old = copy.deepcopy(record)
            old['detection_source'] = 'cix_ssd_300_npu'
            old['vision_model_path'] = '/old/ssd.cix'
            old['vision_tools']['person']['source'] = 'cix_ssd_300_npu'
            updated = agent.with_task_result(old)
            self.assertEqual(updated['vision_model_path'], '/old/ssd.cix')
            self.assertEqual(updated['detection_source'], 'cix_ssd_300_npu')
            self.assertEqual(updated['task_answer'], old['task_answer'])


if __name__ == '__main__':
    unittest.main()
