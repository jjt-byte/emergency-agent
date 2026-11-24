"""YOLO decoding and NPU routing/gates; mock tests do not imply hardware success."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import agent
import agent_loop
from tools import compare_person_npu as comparison
from tools import detect_yolov8_npu as yolo


class YOLOTests(unittest.TestCase):
    def test_person_class_zero_no_objectness_nms_and_stretched_coordinates(self):
        out = np.zeros((1,84,8400), np.float32)
        for idx, box, cls, score in [(0,[320,320,320,320],0,.9),
                                      (1,[322,322,320,320],0,.8),
                                      (2,[100,100,20,40],2,.95),
                                      (3,[50,50,20,20],0,.2),
                                      (4,[100,100,40,40],0,.75)]:
            out[0,:4,idx] = box;out[0,4+cls,idx] = score
        decoded = yolo.decode([out],1280,640,.5,.45)
        self.assertEqual(len(decoded),2)
        self.assertEqual(decoded[0], {'label':'person','confidence':.9,'box':[320.,160.,960.,480.]})
        self.assertEqual(decoded[1]['box'],[160.,80.,240.,120.])
        self.assertEqual(yolo.decode([np.zeros_like(out)],640,640),[])

    def test_bad_output_fails_instead_of_becoming_zero_people(self):
        for value in [np.zeros((1,85,8400)),np.full((1,84,8400),np.nan),np.full((1,84,8400),2.)]:
            with self.subTest(shape=value.shape),self.assertRaises(ValueError):
                yolo.decode([value],640,640)

    def test_fast_preprocessing_keeps_bgr_range_without_external_processes(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'red.png'
            Image.new('RGB', (20, 10), (255, 0, 0)).save(image)
            with patch.object(yolo.subprocess, 'run', side_effect=AssertionError('No per-frame processes')):
                data, size = yolo.prepare_image(image)
        self.assertEqual(size, (20, 10))
        self.assertEqual(data.shape, (1, 3, 640, 640))
        self.assertEqual(data.dtype, np.float32)
        self.assertTrue(data.flags.c_contiguous)
        self.assertTrue(np.all(data[0, :2] == 0))
        self.assertTrue(np.all(data[0, 2] == 1))

    def test_letterbox_diagnosis_restores_original_coordinates(self):
        raw = np.zeros((1, 84, 8400), np.float32)
        raw[0, :4, 0] = [320, 320, 320, 160]
        raw[0, 4, 0] = .9
        result = yolo.decode([raw], 1280, 640, transform={'scale': [.5, .5], 'pad': [0, 160]})
        self.assertEqual(result[0]['box'], [320, 160, 960, 480])

    def test_incomplete_reference_blocks_activation_even_when_counts_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir();config=root/'config.json'
            original=(agent.ROOT/'config.json').read_bytes();config.write_bytes(original)
            def fake_worker(command, **kwargs):
                out=Path(command[command.index('--output')+1]);backend=command[command.index('--worker')+1]
                rows=[{'name':str(n), 'count':n, 'expected_count':n, 'stable':True,
                       'median_seconds':.1, 'reference_complete': n != 4} for n in [6,3,5,4,5,0]]
                out.write_text(json.dumps({'results':rows,'backend':backend}))
                return argparse.Namespace(returncode=0,stdout='',stderr='')
            with patch.object(comparison,'ROOT',root),patch.object(comparison,'REPORT',root/'data/comparison.json'), \
                 patch.object(comparison,'MARKER',root/'data/marker.json'),patch.object(Path,'exists',return_value=True), \
                 patch.object(comparison.subprocess,'run',side_effect=fake_worker),patch.object(comparison,'fingerprint',return_value={'test':'mock-device'}),patch.object(comparison,'render_report'):
                self.assertEqual(comparison.compare(activate=True),3)
            self.assertEqual(config.read_bytes(),original)
            self.assertEqual(json.loads((root/'data/comparison.json').read_text())['status'], 'needs_review')

    def test_npu_dispatch_provenance_and_agent_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);weight=root/'yolov8s.cix';weight.write_bytes(b'model')
            settings={'backend':'cix_yolov8s','python':'/local/noe/python','model_path':str(weight),'threshold':.5}
            payload={'detections':[{'label':'person','confidence':.9,'box':[1,2,20,30]}],
                     'image_size':[100,100], 'source':yolo.SOURCE,'device':'npu'}
            with patch.dict(agent.CONFIG,person_detection=settings),patch.object(agent.subprocess,'run',return_value=argparse.Namespace(returncode=0,stdout=json.dumps(payload))) as run:
                detections,result=agent.run_npu_detector('test.jpg')
                self.assertIn('detect_yolov8_npu.py',run.call_args.args[0][1])
                self.assertEqual(agent.person_source(),yolo.SOURCE)
                self.assertEqual(agent.person_model_path(),weight)
                record={'task':'有几个人','detection_source':yolo.SOURCE,'detections':detections,
                        'vision_result':result,'tool_error':False}
                self.assertEqual(agent.with_task_result(record)['person_statistics']['estimated_count'],1)
                before=agent_loop.fingerprint('detect_people')
                weight.write_bytes(b'changed-model')
                self.assertNotEqual(before,agent_loop.fingerprint('detect_people'))

    def test_no_device_never_activates_or_creates_validation_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.json';original=(agent.ROOT/'config.json').read_bytes();config.write_bytes(original)
            with patch.object(comparison,'ROOT',root),patch.object(comparison,'REPORT',root/'comparison.json'), \
                 patch.object(comparison,'MARKER',root/'marker.json'),patch.object(Path,'exists',return_value=False):
                self.assertEqual(comparison.compare(activate=True),2)
            self.assertEqual(config.read_bytes(),original)
            self.assertEqual(json.loads((root/'comparison.json').read_text())['status'],'blocked')
            self.assertFalse((root/'marker.json').exists())

    def test_incorrect_counts_never_activate_even_if_inference_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir();config=root/'config.json';original=(agent.ROOT/'config.json').read_bytes();config.write_bytes(original)
            def fake_worker(command,**kwargs):
                out=Path(command[command.index('--output')+1]);backend=command[command.index('--worker')+1]
                rows=[{'count':1,'expected_count':6,'stable':True,'median_seconds':.1}]*6
                out.write_text(json.dumps({'results':rows,'backend':backend}))
                return argparse.Namespace(returncode=0,stdout='',stderr='')
            with patch.object(comparison,'ROOT',root),patch.object(comparison,'REPORT',root/'data/comparison.json'), \
                 patch.object(comparison,'MARKER',root/'data/marker.json'),patch.object(Path,'exists',return_value=True), \
                 patch.object(comparison.subprocess,'run',side_effect=fake_worker),patch.object(comparison,'fingerprint',return_value={'test':'mock-device'}),patch.object(comparison,'render_report'):
                self.assertEqual(comparison.compare(activate=True),3)
            self.assertEqual(config.read_bytes(),original)
            self.assertEqual(json.loads((root/'data/marker.json').read_text())['status'],'needs_review')

    def test_successful_comparison_backs_up_and_activates_only_person_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir();config=root/'config.json'
            original=json.loads((agent.ROOT/'config.json').read_text());config.write_text(json.dumps(original))
            def fake_worker(command,**kwargs):
                out=Path(command[command.index('--output')+1]);backend=command[command.index('--worker')+1]
                rows=[{'count':n,'expected_count':n,'stable':True,'median_seconds':.1} for n in [6,3,5,4,5,0]]
                out.write_text(json.dumps({'results':rows,'backend':backend}))
                return argparse.Namespace(returncode=0,stdout='',stderr='')
            with patch.object(comparison,'ROOT',root),patch.object(comparison,'REPORT',root/'data/comparison.json'), \
                 patch.object(comparison,'MARKER',root/'data/marker.json'),patch.object(Path,'exists',return_value=True), \
                 patch.object(comparison.subprocess,'run',side_effect=fake_worker),patch.object(comparison,'fingerprint',return_value={'test':'mock-device'}),patch.object(comparison,'render_report'):
                self.assertEqual(comparison.compare(activate=True),0)
            updated=json.loads(config.read_text())
            self.assertEqual(updated['person_detection']['backend'],'cix_yolov8s')
            self.assertEqual(updated['person_detection_cpu_fallback'],original['person_detection'])
            self.assertEqual(updated['fire_smoke'],original['fire_smoke'])
            self.assertEqual(updated['model_path'],original['model_path'])
            report=json.loads((root/'data/comparison.json').read_text())
            self.assertTrue(report['activated'])
            self.assertEqual(json.loads(Path(report['config_backup']).read_text()),original)


if __name__=='__main__':
    unittest.main()
