"""Preprocessing math and diagnosis: mocks cannot establish NPU success."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from tools.npu_image import resize_bilinear
from tools.diagnose_person_npu import threshold_matrix
from tools import diagnose_person_npu as diagnosis


class DiagnosisTests(unittest.TestCase):
    def test_resize_uses_half_pixel_bilinear_coordinates_and_replicates_edges(self):
        source = np.array([[[255,0,0], [0,255,0]], [[0,0,255], [255,255,255]]], dtype=np.uint8)
        result = resize_bilinear(source, 3, 3)
        np.testing.assert_array_equal(result[0,0], source[0,0])
        np.testing.assert_array_equal(result[-1,-1], source[-1,-1])
        np.testing.assert_array_equal(result[1,1], [128,128,128])
        np.testing.assert_array_equal(resize_bilinear(source, 2, 2), source)

    def test_matrix_distinguishes_low_confidence_and_suppression(self):
        output = np.zeros((1,84,8400), dtype=np.float32)
        output[0,:4,0] = [320,320,200,200];output[0,4,0] = .9
        output[0,:4,1] = [375,320,200,200];output[0,4,1] = .4
        matrix = {(row['threshold'],row['nms_threshold']):row['count']
                  for row in threshold_matrix(output, 640, 640)}
        self.assertEqual(matrix[(.25,.45)], 1)
        self.assertEqual(matrix[(.25,.6)], 2)
        self.assertEqual(matrix[(.5,.6)], 1)

    def test_hardware_failure_preserves_completed_evidence_and_releases_engine(self):
        from PIL import Image
        class Detector:
            closed = False

            def infer_raw(self, image, profile):
                if Path(image).name == 'second.png':
                    raise RuntimeError('driver interrupted')
                return np.zeros((1, 84, 8400), np.float32), {
                    'image_size': [8, 8], 'transform': {'scale': [80, 80], 'pad': [0, 0]},
                    'preprocess_seconds': .01, 'forward_seconds': .02}

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            Image.new('RGB', (8, 8)).save(directory/'first.png')
            cases = [{'name': name, 'image': directory/(name+'.png'), 'expected_count': 0}
                     for name in ['first', 'second']]
            detector = Detector()
            rows = []
            with patch.object(diagnosis.yolo, 'LocalDetector', return_value=detector), \
                 patch.object(diagnosis, 'PROFILES', ['bgr_stretch']), \
                 self.assertRaisesRegex(RuntimeError, 'driver interrupted'):
                diagnosis.hardware_checks(cases, directory, 1, rows=rows)
            self.assertTrue(detector.closed)
            self.assertEqual(len(rows), 1)
            evidence = json.loads((directory/'first-bgr_stretch.json').read_text())
            self.assertEqual(evidence['count'], 0)
            self.assertTrue((directory/evidence['raw_output_file']).is_file())
            self.assertTrue((directory/evidence['overlay_file']).is_file())


if __name__ == '__main__':
    unittest.main()
