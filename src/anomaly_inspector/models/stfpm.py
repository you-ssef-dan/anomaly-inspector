import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

from anomaly_inspector.models.backbones import build_backbone

LAYERS = ("layer1", "layer2", "layer3")
Features = dict[str, torch.Tensor]


class STFPM(nn.Module):
    """STFPM (Wang et al., 2021): a student network learns to copy a frozen teacher.

    The student only ever practises on good images. Where it fails to reproduce the teacher's
    features, the image differs from anything it saw: that gap, layer by layer, is the anomaly map.
    """

    def __init__(
        self, teacher: nn.Module, student: nn.Module, layers: tuple[str, ...] = LAYERS
    ) -> None:
        super().__init__()
        nodes = {name: name for name in layers}
        self.teacher: nn.Module = create_feature_extractor(teacher, nodes)
        self.teacher.eval().requires_grad_(False)
        self.student: nn.Module = create_feature_extractor(student, nodes)
        self.layers = layers

    @classmethod
    def from_backbone(cls, name: str = "resnet18", pretrained: bool = True) -> "STFPM":
        """An ImageNet teacher and a randomly initialized student of the same architecture."""
        teacher = build_backbone(name, pretrained=pretrained)
        return cls(teacher, build_backbone(name, pretrained=False))

    def train(self, mode: bool = True) -> "STFPM":
        super().train(mode)
        self.teacher.eval()  # otherwise its BatchNorm statistics drift while the student trains
        return self

    def forward(self, x: torch.Tensor) -> tuple[Features, Features]:
        """Teacher and student features for each layer; only the student builds a graph."""
        with torch.no_grad():
            teacher = self.teacher(x)
        return teacher, self.student(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the anomaly map (B, 1, H, W) and the image score (B,), the map's maximum."""
        if self.student.training:
            raise RuntimeError("call model.eval() before predict(), or BatchNorm keeps updating")
        teacher, student = self(x)
        anomaly_map = stfpm_anomaly_map(teacher, student, (x.shape[-2], x.shape[-1]))
        return anomaly_map, anomaly_map.flatten(1).amax(dim=1)


def layer_distance(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """Half the squared distance between channel-normalized features: (B, 1, h, w), in [0, 2]."""
    gap = F.normalize(teacher, dim=1) - F.normalize(student, dim=1)
    return 0.5 * gap.pow(2).sum(1, keepdim=True)


def stfpm_loss(teacher: Features, student: Features) -> torch.Tensor:
    """Sum over layers of the mean distance (paper, equation 2)."""
    return torch.stack([layer_distance(teacher[n], student[n]).mean() for n in teacher]).sum()


def stfpm_anomaly_map(teacher: Features, student: Features, size: tuple[int, int]) -> torch.Tensor:
    """Each layer's distance map upsampled to image size, multiplied across layers (paper, 3.3)."""
    maps = [
        F.interpolate(layer_distance(teacher[n], student[n]), size=size, mode="bilinear")
        for n in teacher
    ]
    return torch.stack(maps).prod(dim=0)
