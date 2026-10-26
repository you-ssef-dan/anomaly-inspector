from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    wide_resnet50_2,
)

BACKBONES = ["wide_resnet50_2", "resnet18"]


def build_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """An ImageNet backbone by name. pretrained=False gives random weights and downloads nothing."""
    model: nn.Module
    if name == "wide_resnet50_2":
        # The PatchCore paper used the original torchvision weights (V1), not the newer V2.
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        model = wide_resnet50_2(weights=weights)
    elif name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"unknown backbone {name!r}, expected one of {BACKBONES}")
    return model
