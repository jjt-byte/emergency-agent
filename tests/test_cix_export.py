"""CIX candidate export contract, separate from compilation or NPU validation."""
from pathlib import Path
import tempfile
import unittest

from tools.export_damo_cix import build_cfg, fixed_pixels, image_list


class ExportContractTests(unittest.TestCase):
    def test_fixed_input_preserves_rgb_range_and_padding(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest('Run preprocessing checks with .venv-fire/bin/python')
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'red.png'
            Image.new('RGB', (1280, 320), color=(255, 0, 0)).save(image)
            pixels, scales = fixed_pixels(image)
        self.assertEqual(pixels.shape, (1, 3, 640, 640))
        self.assertEqual(pixels.dtype, np.float32)
        self.assertEqual(scales, (.5, .5))
        self.assertTrue(np.all(pixels[0, 0, :160] == 255))
        self.assertFalse(np.any(pixels[0, 1:]))
        self.assertFalse(np.any(pixels[:, :, 160:]))

    def test_small_images_are_not_upscaled(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Run preprocessing checks with .venv-fire/bin/python')
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'small.png'
            Image.new('RGB', (10, 6), color=(0, 255, 0)).save(image)
            pixels, scales = fixed_pixels(image)
        self.assertEqual(scales, (1, 1))
        self.assertEqual(float(pixels[0, 1, 5, 9]), 255)
        self.assertEqual(float(pixels[0, 1, 6, 10]), 0)

    def test_voc_list_paths_resolve_and_missing_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'image.jpg').write_bytes(b'placeholder')
            listing = root/'train.txt'
            listing.write_text('# comment\nimage.jpg annotations/image.xml\nimage.jpg\n')
            self.assertEqual(image_list(listing), [root/'image.jpg'])
            listing.write_text('missing.jpg missing.xml\n')
            with self.assertRaisesRegex(ValueError, '不存在'):
                image_list(listing)

    def test_compiler_cfg_leaves_nms_on_cpu_and_uses_graph_names(self):
        cfg = build_cfg('damo_fire_smoke', 'images.1', 'output.1', 640)
        self.assertIn('model_type = torch', cfg)
        self.assertIn('input = images.1', cfg)
        self.assertIn('output = output.1', cfg)
        self.assertIn('target = X2_1204MP3', cfg)
        self.assertIn('detection_postprocess =\n', cfg)
        self.assertNotIn('trigger_float_op', cfg)


if __name__ == '__main__':
    unittest.main()
