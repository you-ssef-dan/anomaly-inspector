"""storage.py: a saved PatchCore comes back identical, from the file alone."""

from pathlib import Path

import pytest
import torch

from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore
from anomaly_inspector.models.storage import load_patchcore, save_patchcore


def fitted(seed: int = 0) -> PatchCore:
    torch.manual_seed(seed)
    model = PatchCore(build_backbone("resnet18", pretrained=False))
    images = torch.rand(4, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    model.fit([(images, torch.zeros(4))], coreset_ratio=0.5, seed=0)
    return model.eval()


def test_a_saved_model_scores_exactly_like_the_original(tmp_path: Path) -> None:
    model = fitted()
    # 72: a size no default uses, so the file has to be the one carrying it.
    save_patchcore(model, "resnet18", 72, tmp_path / "deep" / "model.pt")

    saved = load_patchcore(tmp_path / "deep" / "model.pt")

    assert (saved.backbone, saved.image_size) == ("resnet18", 72)
    assert not saved.model.training
    images = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(2))
    for before, after in zip(model(images), saved.model(images), strict=True):
        assert torch.equal(before, after)


def test_the_same_model_always_gives_the_same_bytes(tmp_path: Path) -> None:
    # DVC compares files by hash: identical bytes let it skip the stages after training.
    # Same file name on purpose: torch.save writes the name into the archive it creates.
    first = save_patchcore(fitted(), "resnet18", 64, tmp_path / "a" / "model.pt").read_bytes()
    second = save_patchcore(fitted(), "resnet18", 64, tmp_path / "b" / "model.pt").read_bytes()
    assert first == second


def test_loading_downloads_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = save_patchcore(fitted(), "resnet18", 64, tmp_path / "model.pt")
    asked: list[bool] = []
    real = build_backbone

    def spy(name: str, pretrained: bool = True) -> torch.nn.Module:
        asked.append(pretrained)
        return real(name, pretrained=pretrained)

    monkeypatch.setattr("anomaly_inspector.models.storage.build_backbone", spy)
    load_patchcore(path)
    assert asked == [False]  # the weights come from the file
