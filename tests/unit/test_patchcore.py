from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.data.manifest import build_manifest
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import (
    PatchCore,
    gaussian_kernel,
    greedy_coreset,
    nearest_distance,
)

SIZE = 64


def tiny_model(seed: int = 0) -> PatchCore:
    """PatchCore on an untrained ResNet-18: fast, and nothing to download."""
    torch.manual_seed(seed)
    return PatchCore(build_backbone("resnet18", pretrained=False))


def good_images(n: int, seed: int = 0) -> torch.Tensor:
    """A smooth diagonal ramp plus a little noise: the 'normal part' of these tests."""
    g = torch.Generator().manual_seed(seed)
    ramp = torch.linspace(-1, 1, SIZE)
    base = (ramp[None, :] + ramp[:, None]).expand(3, SIZE, SIZE) * 0.5
    return torch.stack([base + 0.05 * torch.randn(3, SIZE, SIZE, generator=g) for _ in range(n)])


def with_defect(images: torch.Tensor) -> torch.Tensor:
    """The same images with a bright 12 x 12 square at rows and columns 8 to 20."""
    defective = images.clone()
    defective[:, :, 8:20, 8:20] = 3.0
    return defective


def fitted(ratio: float = 1.0, seed: int = 0) -> PatchCore:
    model = tiny_model()
    model.fit([(good_images(8), torch.zeros(8))], coreset_ratio=ratio, seed=seed)
    return model


# --- the model ---------------------------------------------------------------------------


def test_embed_gives_one_feature_vector_per_patch() -> None:
    features = tiny_model().embed(good_images(2))
    # ResNet-18: layer2 has 128 channels, layer3 256; layer2's grid is 1/8 of the image
    assert features.shape == (2, 128 + 256, SIZE // 8, SIZE // 8)


def test_embed_is_locally_averaged_layer2_plus_upsampled_layer3() -> None:
    # Paper, section 3.1: each patch is the 3 x 3 average of its neighbourhood ("locally aware"),
    # and the coarser layer3 grid is upsampled to layer2's grid before concatenating.
    model = tiny_model()
    images = good_images(2)
    raw = model.extractor(images)
    layer2 = F.avg_pool2d(raw["layer2"], 3, stride=1, padding=1)
    layer3 = F.avg_pool2d(raw["layer3"], 3, stride=1, padding=1)
    layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear")
    torch.testing.assert_close(model.embed(images), torch.cat([layer2, layer3], dim=1))


def test_forward_follows_the_patchcore_definition() -> None:
    # Image score = the largest patch distance; anomaly map = the patch distances upsampled
    # bilinearly to image size, then smoothed with a Gaussian of sigma 4.
    model = fitted(ratio=0.5)
    images = with_defect(good_images(2, seed=5))
    features = model.embed(images)
    b, c, h, w = features.shape
    patch_distances = nearest_distance(
        features.permute(0, 2, 3, 1).reshape(-1, c), model.memory_bank
    ).reshape(b, 1, h, w)
    upsampled = F.interpolate(patch_distances, size=(SIZE, SIZE), mode="bilinear")
    smoothed = F.conv2d(F.pad(upsampled, (16, 16, 16, 16), mode="reflect"), gaussian_kernel(4.0))

    anomaly_map, scores = model(images)
    torch.testing.assert_close(scores, patch_distances.flatten(1).amax(dim=1))
    torch.testing.assert_close(anomaly_map, smoothed)


def test_fit_keeps_the_requested_share_of_patches() -> None:
    model = fitted(ratio=0.1)
    all_patches = 8 * (SIZE // 8) ** 2
    assert model.memory_bank.shape == (int(all_patches * 0.1), 384)


def test_forward_returns_a_full_size_map_and_one_score_per_image() -> None:
    anomaly_map, scores = fitted()(good_images(3, seed=1))
    assert anomaly_map.shape == (3, 1, SIZE, SIZE)
    assert scores.shape == (3,)


def test_training_images_score_near_zero_when_every_patch_is_kept() -> None:
    _, scores = fitted(ratio=1.0)(good_images(8))
    assert scores.max() < 0.05  # each patch is in the bank: distance 0 up to rounding


def test_a_defect_scores_far_higher_than_new_good_images() -> None:
    model = fitted()
    _, good = model(good_images(4, seed=1))
    _, bad = model(with_defect(good_images(4, seed=1)))
    assert bad.min() > 10 * good.max()


def test_the_anomaly_map_is_highest_on_the_defect() -> None:
    anomaly_map, _ = fitted()(with_defect(good_images(4, seed=2)))
    on_defect = anomaly_map[:, 0, 8:20, 8:20].mean()
    far_away = anomaly_map[:, 0, 32:, 32:].mean()
    assert on_defect > 2 * far_away
    for one_map in anomaly_map[:, 0]:
        y, x = divmod(int(one_map.argmax()), SIZE)
        # on the square (rows and columns 8 to 20), give or take the blur
        assert 0 <= y <= 28
        assert 0 <= x <= 28


def test_scores_do_not_depend_on_the_rest_of_the_batch() -> None:
    model = fitted(ratio=0.5)
    images = good_images(4, seed=3)
    _, batch = model(images)
    one_by_one = torch.cat([model(images[i : i + 1])[1] for i in range(4)])
    torch.testing.assert_close(batch, one_by_one, atol=1e-3, rtol=0)


def test_same_seed_gives_the_same_memory_bank() -> None:
    assert torch.equal(fitted(0.1, seed=0).memory_bank, fitted(0.1, seed=0).memory_bank)
    assert not torch.equal(fitted(0.1, seed=0).memory_bank, fitted(0.1, seed=1).memory_bank)


def test_forward_before_fit_explains_the_problem() -> None:
    with pytest.raises(RuntimeError, match="call fit"):
        tiny_model()(good_images(1))


def test_fit_refuses_defective_images() -> None:
    with pytest.raises(ValueError, match="good images only"):
        tiny_model().fit([(good_images(2), torch.tensor([0, 1]))], coreset_ratio=1.0)


def test_backbone_stays_frozen_even_after_train() -> None:
    model = tiny_model().train()
    assert model.training
    assert not model.extractor.training
    assert not any(m.training for m in model.extractor.modules())
    assert not any(p.requires_grad for p in model.parameters())


def test_a_saved_model_loads_into_a_fresh_one() -> None:
    model = fitted(ratio=0.5)
    fresh = tiny_model(seed=123)  # different random weights and an empty bank
    fresh.load_state_dict(model.state_dict())
    images = with_defect(good_images(2, seed=4))
    for a, b in zip(model(images), fresh(images), strict=True):
        torch.testing.assert_close(a, b)


def test_fits_from_a_dataloader_over_manifest_rows(fake_mvtec: Path) -> None:
    manifest = build_manifest(fake_mvtec, ["bottle"])
    train = ManifestDataset(manifest[manifest["source_split"] == "train"], SIZE)
    test = ManifestDataset(manifest[manifest["source_split"] == "test"], SIZE)
    model = tiny_model()
    model.fit(DataLoader(train, batch_size=2), coreset_ratio=0.5)
    images, _ = next(iter(DataLoader(test, batch_size=len(test))))
    anomaly_map, scores = model(images)
    assert anomaly_map.shape == (len(test), 1, SIZE, SIZE)
    assert torch.isfinite(scores).all()


# --- the building blocks -----------------------------------------------------------------


def test_nearest_distance_matches_brute_force() -> None:
    g = torch.Generator().manual_seed(0)
    patches, bank = torch.randn(50, 16, generator=g), torch.randn(20, 16, generator=g)
    expected = torch.cdist(patches, bank).amin(dim=1)
    torch.testing.assert_close(nearest_distance(patches, bank), expected)


def test_coreset_has_k_rows_taken_from_the_input() -> None:
    features = torch.randn(100, 8, generator=torch.Generator().manual_seed(0))
    coreset = greedy_coreset(features, ratio=0.1)
    assert coreset.shape == (10, 8)
    assert all((features == row).all(dim=1).any() for row in coreset)
    assert greedy_coreset(features, ratio=0.001).shape == (1, 8)  # never empty


def test_coreset_with_ratio_one_keeps_every_row_once() -> None:
    features = torch.randn(30, 8, generator=torch.Generator().manual_seed(0))
    coreset = greedy_coreset(features, ratio=1.0)
    assert sorted(coreset[:, 0].tolist()) == sorted(features[:, 0].tolist())


def test_coreset_covers_every_cluster() -> None:
    # Four far-apart clusters and room for four points: k-center greedy takes one from each,
    # where a random pick of four would often miss a cluster.
    g = torch.Generator().manual_seed(0)
    centers = torch.eye(4, 16) * 50
    features = torch.cat([c + torch.randn(25, 16, generator=g) for c in centers])
    cluster_of = torch.cdist(greedy_coreset(features, ratio=0.04), centers).argmin(dim=1)
    assert sorted(cluster_of.tolist()) == [0, 1, 2, 3]


def test_gaussian_kernel_is_normalized_symmetric_and_peaked() -> None:
    kernel = gaussian_kernel(sigma=4.0)
    assert kernel.shape == (1, 1, 33, 33)  # radius int(4 * 4 + 0.5) = 16
    torch.testing.assert_close(kernel.sum(), torch.tensor(1.0))
    torch.testing.assert_close(kernel, kernel.flip(-1))
    torch.testing.assert_close(kernel, kernel.transpose(-1, -2))
    assert kernel[0, 0, 16, 16] == kernel.max()


@pytest.mark.parametrize("name", ["wide_resnet50_2", "resnet18"])
def test_backbones_build_without_downloading(name: str) -> None:
    backbone = build_backbone(name, pretrained=False)
    assert hasattr(backbone, "layer2")
    assert hasattr(backbone, "layer3")


def test_unknown_backbone_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backbone"):
        build_backbone("vgg16")
