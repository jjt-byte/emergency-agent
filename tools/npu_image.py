"""In-process RGB decoding and half-pixel bilinear resize for NPU adapters."""
import numpy as np


VERSION = 'pillow_numpy_half_pixel_bilinear_v1'


def resize_bilinear(pixels, width, height):
    """Resize uint8 HWC pixels with half-pixel coordinates and replicated edges.

    This follows INTER_LINEAR geometry without the FFmpeg scaler's downsample
    antialiasing. Integer rounding can differ from OpenCV by one pixel level.
    """
    source_height, source_width = pixels.shape[:2]
    if (source_width, source_height) == (width, height):
        return np.ascontiguousarray(pixels)
    xs = (np.arange(width, dtype=np.float32)+.5)*(source_width/width)-.5
    ys = (np.arange(height, dtype=np.float32)+.5)*(source_height/height)-.5
    xs = np.clip(xs, 0, source_width-1)
    ys = np.clip(ys, 0, source_height-1)
    x0, y0 = xs.astype(np.int32), ys.astype(np.int32)
    x1, y1 = np.minimum(x0+1, source_width-1), np.minimum(y0+1, source_height-1)
    wx = (xs-x0.astype(np.float32))[None, :, None]
    wy = (ys-y0.astype(np.float32))[:, None, None]
    upper = pixels[y0[:, None], x0[None, :]]*(1-wx) + pixels[y0[:, None], x1[None, :]]*wx
    lower = pixels[y1[:, None], x0[None, :]]*(1-wx) + pixels[y1[:, None], x1[None, :]]*wx
    return np.clip(np.floor(upper*(1-wy)+lower*wy+.5), 0, 255).astype(np.uint8)


def prepare(image, color='BGR', geometry='stretch'):
    """Return NCHW float32 0..1 input, original size and resize geometry."""
    from PIL import Image
    with Image.open(image) as opened:
        rgb = np.array(opened.convert('RGB'), dtype=np.uint8)
    height, width = rgb.shape[:2]
    if color not in {'RGB', 'BGR'} or geometry not in {'stretch', 'letterbox'}:
        raise ValueError('不支持的 NPU 图片预处理方式')
    if geometry == 'stretch':
        resized = resize_bilinear(rgb, 640, 640)
        metadata = {'scale': [640/width, 640/height], 'pad': [0, 0]}
    else:
        scale = min(640/width, 640/height)
        target_width, target_height = max(1, round(width*scale)), max(1, round(height*scale))
        resized = np.full((640, 640, 3), 114, dtype=np.uint8)
        left, top = (640-target_width)//2, (640-target_height)//2
        resized[top:top+target_height, left:left+target_width] = resize_bilinear(rgb, target_width, target_height)
        metadata = {'scale': [target_width/width, target_height/height], 'pad': [left, top]}
    if color == 'BGR':
        resized = resized[:, :, ::-1]
    tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.
    return tensor, (width, height), metadata
