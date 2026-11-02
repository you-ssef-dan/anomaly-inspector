"""Brightness, contrast and sharpness of a photo: what new light, a moved camera or a dirty lens
change first, often before the model's scores do.

Measured the same way on the reference images at training time and on the line's photos in
production, so the two can be compared. numpy and Pillow only, like preprocessing.py.
"""

import numpy as np
from PIL import Image


def image_stats(image: Image.Image) -> dict[str, float]:
    """On the photo in gray levels (0 to 255): brightness is their mean, contrast their standard
    deviation, and sharpness the variance of their Laplacian, which drops when the photo blurs."""
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    # The Laplacian: each pixel's four neighbours minus four times itself (the border is left out).
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    )
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(laplacian.var()),
    }
