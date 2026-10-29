"""INT8 version of an ONNX PatchCore: weights and activations of the convolutions in 8 bits.

Only the convolutions (the backbone) are quantized. The distances to the memory bank stay in
FP32: a small error there would move scores across the threshold and flip decisions.
"""

import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class Images(CalibrationDataReader):  # type: ignore[misc]  # onnxruntime has no type hints
    """Feeds calibration images (C, H, W) one at a time, as ONNX Runtime asks for them."""

    def __init__(self, images: Iterable[NDArray[np.float32]]) -> None:
        self.images: Iterator[NDArray[np.float32]] = iter(images)

    def get_next(self) -> dict[str, NDArray[np.float32]] | None:
        image = next(self.images, None)
        return None if image is None else {"image": image[None]}


def quantize_int8(fp32: Path, int8: Path, images: Iterable[NDArray[np.float32]]) -> Path:
    """Write the INT8 model. The images set the activations' ranges: use good train images."""
    with tempfile.TemporaryDirectory() as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        # Graph optimisation before quantization. Symbolic shape inference is skipped: it
        # fails on the graphs torch.export writes, and the shapes are already in the file.
        quant_pre_process(str(fp32), str(prepared), skip_symbolic_shape=True)
        quantize_static(
            str(prepared),
            str(int8),
            calibration_data_reader=Images(images),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            op_types_to_quantize=["Conv"],
            # Merge the value ranges every 16 images instead of keeping every image's
            # intermediate outputs in memory.
            extra_options={"CalibMaxIntermediateOutputs": 16},
        )
    return int8
