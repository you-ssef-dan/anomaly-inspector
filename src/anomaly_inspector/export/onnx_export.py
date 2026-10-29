"""PatchCore to ONNX: one self-contained file that ONNX Runtime can run without PyTorch.

The file also carries what serving needs (threshold, image size, category, git SHA) as metadata,
so the API reads them from the model instead of hard-coding them.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


def export_onnx(model: nn.Module, path: Path, size: int) -> Path:
    """Write the model as one ONNX file with a dynamic batch size."""
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.randn(2, 3, size, size),),  # batch 2: torch.export treats a size-1 dim as fixed
        str(path),
        input_names=["image"],
        output_names=["anomaly_map", "score"],
        dynamic_shapes={"x": {0: "batch"}},  # "x" is the name of forward()'s argument
        external_data=False,  # one file, weights included (fine below 2 GB)
    )
    return path


def write_metadata(path: Path, metadata: dict[str, str]) -> None:
    """Check the file is a valid ONNX model, then store the metadata inside it."""
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    onnx.helper.set_model_props(model, metadata)
    onnx.save(model, str(path))


def read_metadata(path: Path) -> dict[str, str]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return dict(session.get_modelmeta().custom_metadata_map)


class OnnxPatchCore:
    """An ONNX model behind the same call as PatchCore: images -> (anomaly maps, scores).

    So the evaluation code scores it exactly as it scores the PyTorch model.
    """

    def __init__(self, path: Path) -> None:
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        anomaly_map, score = self.session.run(None, {"image": images.numpy().astype(np.float32)})
        return torch.from_numpy(anomaly_map), torch.from_numpy(score)
