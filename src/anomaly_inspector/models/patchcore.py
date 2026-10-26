from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor


class PatchCore(nn.Module):
    """PatchCore (Roth et al., CVPR 2022): a memory bank of patch features from good images.

    Nothing is trained. fit() runs the frozen backbone once over good images and keeps a coreset
    of their patch features; at test time a patch far from every stored patch is anomalous.
    """

    memory_bank: torch.Tensor
    blur: torch.Tensor

    def __init__(
        self,
        backbone: nn.Module,
        layers: tuple[str, ...] = ("layer2", "layer3"),
        sigma: float = 4.0,
    ) -> None:
        super().__init__()
        self.extractor = create_feature_extractor(backbone, {name: name for name in layers})
        self.extractor.eval().requires_grad_(False)
        self.layers = layers
        self.register_buffer("memory_bank", torch.empty(0, 0))
        self.register_buffer("blur", gaussian_kernel(sigma))

    def train(self, mode: bool = True) -> "PatchCore":
        super().train(mode)
        self.extractor.eval()  # frozen backbone: BatchNorm must keep its ImageNet statistics
        return self

    def _load_from_state_dict(
        self, state_dict: dict[str, Any], prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        # A new PatchCore starts with an empty (0, 0) bank; resize it so a saved bank can load.
        saved = state_dict.get(prefix + "memory_bank")
        if saved is not None:
            self.memory_bank = torch.empty_like(saved)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) images -> (B, C, h, w) locally aware patch features."""
        features = self.extractor(x)
        maps = [F.avg_pool2d(features[name], 3, stride=1, padding=1) for name in self.layers]
        size = maps[0].shape[-2:]
        maps = [maps[0], *(F.interpolate(m, size=size, mode="bilinear") for m in maps[1:])]
        return torch.cat(maps, dim=1)

    @torch.no_grad()
    def fit(
        self,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        coreset_ratio: float,
        seed: int = 0,
    ) -> None:
        """Keep a coreset of the patch features of good images (label 0 only)."""
        patches = []
        for x, labels in loader:
            if bool((labels != 0).any()):
                raise ValueError("fit() takes good images only (label 0)")
            patches.append(self.embed(x).permute(0, 2, 3, 1).flatten(0, 2))
        self.memory_bank = greedy_coreset(torch.cat(patches), coreset_ratio, seed=seed)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the anomaly map (B, 1, H, W) and the image score (B,)."""
        if self.memory_bank.numel() == 0:
            raise RuntimeError("PatchCore has no memory bank yet: call fit() first")
        features = self.embed(x)
        b, c, h, w = features.shape
        distances = nearest_distance(features.permute(0, 2, 3, 1).reshape(-1, c), self.memory_bank)
        patch_scores = distances.reshape(b, 1, h, w)
        anomaly_map = F.interpolate(patch_scores, size=x.shape[-2:], mode="bilinear")
        p = self.blur.shape[-1] // 2
        anomaly_map = F.conv2d(F.pad(anomaly_map, (p, p, p, p), mode="reflect"), self.blur)
        return anomaly_map, patch_scores.flatten(1).amax(dim=1)


def nearest_distance(patches: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Euclidean distance from each patch (N, C) to its nearest bank entry (M, C).

    |a - b|^2 = |a|^2 - 2 a.b + |b|^2 as one matrix product: no torch.cdist, so it exports to ONNX.
    """
    d2 = patches.pow(2).sum(1, keepdim=True) - 2 * patches @ bank.T + bank.pow(2).sum(1)
    return d2.clamp_min(0).amin(dim=1).sqrt()


def greedy_coreset(
    features: torch.Tensor, ratio: float, dim: int = 128, seed: int = 0
) -> torch.Tensor:
    """k-center greedy selection on a random projection (paper, section 3.2).

    Each step adds the patch farthest from everything chosen so far, so a small coreset still
    covers every kind of normal patch.
    """
    g = torch.Generator().manual_seed(seed)
    z = features @ (torch.randn(features.shape[1], dim, generator=g) / dim**0.5)
    k = max(1, int(len(features) * ratio))
    chosen = [int(torch.randint(len(features), (1,), generator=g))]
    min_d = (z - z[chosen[0]]).pow(2).sum(1)
    for _ in range(k - 1):
        chosen.append(int(min_d.argmax()))
        min_d = torch.minimum(min_d, (z - z[chosen[-1]]).pow(2).sum(1))
    return features[chosen]


def gaussian_kernel(sigma: float) -> torch.Tensor:
    """A normalized 2D Gaussian of shape (1, 1, 2r + 1, 2r + 1), r = 4 sigma, for smoothing maps."""
    r = int(4 * sigma + 0.5)
    t = torch.exp(-(torch.arange(-r, r + 1, dtype=torch.float32) ** 2) / (2 * sigma**2))
    t = t / t.sum()
    return (t[:, None] * t[None, :])[None, None]
