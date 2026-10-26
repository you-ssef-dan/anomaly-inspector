import numpy as np
from numpy.typing import NDArray
from PIL import Image

# ImageNet statistics: the pretrained backbones expect inputs normalized with these.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image, size: int) -> NDArray[np.float32]:
    """Convert to RGB, resize to size x size, normalize. Returns float32 of shape (3, size, size).

    Training and serving both call this, so they cannot drift apart. It needs only numpy and
    Pillow, so the API image does not need PyTorch. The aspect ratio is not kept; MVTec AD
    images are square.
    """
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.ascontiguousarray(((x - MEAN) / STD).transpose(2, 0, 1), dtype=np.float32)
