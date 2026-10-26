import copy

import pytest
import torch
import torch.nn.functional as F

from anomaly_inspector.models import stfpm as stfpm_module
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.stfpm import (
    STFPM,
    layer_distance,
    stfpm_anomaly_map,
    stfpm_loss,
)

SIZE = 64


def tiny_model(seed: int = 0) -> STFPM:
    """Untrained ResNet-18 teacher and student: fast, and nothing to download."""
    torch.manual_seed(seed)
    return STFPM.from_backbone("resnet18", pretrained=False)


def good_images(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ramp = torch.linspace(-1, 1, SIZE)
    base = (ramp[None, :] + ramp[:, None]).expand(3, SIZE, SIZE) * 0.5
    return torch.stack([base + 0.05 * torch.randn(3, SIZE, SIZE, generator=g) for _ in range(n)])


def with_defect(images: torch.Tensor) -> torch.Tensor:
    defective = images.clone()
    defective[:, :, 8:20, 8:20] = 3.0
    return defective


def train_student(model: STFPM, steps: int = 40) -> list[float]:
    """The guide's optimizer settings, on 8 good images."""
    optimizer = torch.optim.SGD(model.student.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)
    images, losses = good_images(8), []
    model.train()
    for _ in range(steps):
        loss = stfpm_loss(*model(images))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@pytest.fixture(scope="module")
def trained() -> tuple[STFPM, list[float], dict[str, torch.Tensor]]:
    """One short training run shared by several tests (about 3 seconds)."""
    model = tiny_model()
    teacher_before = copy.deepcopy(model.teacher.state_dict())
    losses = train_student(model)
    return model.eval(), losses, teacher_before


# --- the model ---------------------------------------------------------------------------


def test_forward_gives_teacher_and_student_features_for_each_layer() -> None:
    teacher, student = tiny_model()(good_images(2))
    expected = {"layer1": (2, 64, 16, 16), "layer2": (2, 128, 8, 8), "layer3": (2, 256, 4, 4)}
    assert {name: tuple(f.shape) for name, f in teacher.items()} == expected
    assert {name: tuple(f.shape) for name, f in student.items()} == expected


def test_only_the_student_can_learn() -> None:
    model = tiny_model()
    assert not any(p.requires_grad for p in model.teacher.parameters())
    assert all(p.requires_grad for p in model.student.parameters())


def test_teacher_stays_in_eval_mode_when_the_model_trains() -> None:
    model = tiny_model().train()
    assert model.student.training
    assert not any(m.training for m in model.teacher.modules())


def test_the_loss_sends_gradients_to_the_student_only() -> None:
    model = tiny_model().train()
    stfpm_loss(*model(good_images(2))).backward()
    assert all(p.grad is not None for p in model.student.parameters())
    assert all(p.grad is None for p in model.teacher.parameters())


def test_training_lowers_the_loss(trained: tuple[STFPM, list[float], dict]) -> None:
    _, losses, _ = trained
    assert losses[-1] < 0.25 * losses[0]


def test_training_never_changes_the_teacher(trained: tuple[STFPM, list[float], dict]) -> None:
    model, _, teacher_before = trained
    for name, value in model.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)


def test_a_defect_scores_far_higher_after_training(
    trained: tuple[STFPM, list[float], dict],
) -> None:
    model, _, _ = trained
    _, good = model.predict(good_images(4, seed=1))
    anomaly_map, bad = model.predict(with_defect(good_images(4, seed=1)))
    assert bad.min() > 5 * good.max()
    assert anomaly_map[:, 0, 8:20, 8:20].mean() > 10 * anomaly_map[:, 0, 40:, 40:].mean()


def test_a_student_identical_to_the_teacher_sees_nothing_wrong() -> None:
    # The whole idea: an anomaly is wherever the student cannot reproduce the teacher.
    model = tiny_model()
    model.student.load_state_dict(model.teacher.state_dict())
    model.eval()
    images = with_defect(good_images(2))
    assert stfpm_loss(*model(images)).item() == pytest.approx(0, abs=1e-6)
    anomaly_map, scores = model.predict(images)
    assert anomaly_map.abs().max().item() == pytest.approx(0, abs=1e-6)
    assert scores.abs().max().item() == pytest.approx(0, abs=1e-6)


def test_predict_returns_a_full_size_map_and_its_maximum_as_score(
    trained: tuple[STFPM, list[float], dict],
) -> None:
    model, _, _ = trained
    anomaly_map, scores = model.predict(with_defect(good_images(3, seed=2)))
    assert anomaly_map.shape == (3, 1, SIZE, SIZE)
    torch.testing.assert_close(scores, anomaly_map.flatten(1).amax(dim=1))


def test_predict_refuses_to_run_in_training_mode() -> None:
    with pytest.raises(RuntimeError, match=r"model\.eval\(\)"):
        tiny_model().train().predict(good_images(1))


def test_a_saved_model_loads_into_a_fresh_one(trained: tuple[STFPM, list[float], dict]) -> None:
    model, _, _ = trained
    fresh = tiny_model(seed=123)
    fresh.load_state_dict(model.state_dict())
    fresh.eval()
    images = with_defect(good_images(2, seed=3))
    for a, b in zip(model.predict(images), fresh.predict(images), strict=True):
        torch.testing.assert_close(a, b)


def test_from_backbone_asks_for_a_pretrained_teacher_and_a_random_student(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def spy(name: str, pretrained: bool = True) -> torch.nn.Module:
        calls.append((name, pretrained))
        return build_backbone(name, pretrained=False)  # no download in tests

    monkeypatch.setattr(stfpm_module, "build_backbone", spy)
    STFPM.from_backbone("resnet18", pretrained=True)
    assert calls == [("resnet18", True), ("resnet18", False)]


def test_teacher_comes_pretrained_and_student_starts_random() -> None:
    model = tiny_model()
    teacher_weights = model.teacher.state_dict()
    student_weights = model.student.state_dict()
    assert teacher_weights.keys() == student_weights.keys()  # same architecture
    first_conv = next(k for k in teacher_weights if k.endswith("conv1.weight"))
    assert not torch.equal(teacher_weights[first_conv], student_weights[first_conv])


# --- the building blocks -----------------------------------------------------------------


def test_layer_distance_hand_computed_cases() -> None:
    t = torch.randn(2, 8, 4, 4, generator=torch.Generator().manual_seed(0))
    zero = torch.zeros(2, 1, 4, 4)
    torch.testing.assert_close(layer_distance(t, t), zero)  # identical
    torch.testing.assert_close(layer_distance(t, 3 * t), zero)  # only the direction counts
    torch.testing.assert_close(layer_distance(t, -t), zero + 2)  # opposite: 0.5 * |2u|^2 = 2


def test_loss_is_the_sum_over_layers_of_the_mean_distance() -> None:
    g = torch.Generator().manual_seed(0)
    teacher = {n: torch.randn(2, 4, s, s, generator=g) for n, s in [("a", 8), ("b", 4)]}
    student = {n: torch.randn_like(f) for n, f in teacher.items()}
    expected = sum(layer_distance(teacher[n], student[n]).mean() for n in teacher)
    torch.testing.assert_close(stfpm_loss(teacher, student), expected)


def test_anomaly_map_multiplies_the_upsampled_layer_maps() -> None:
    g = torch.Generator().manual_seed(0)
    teacher = {n: torch.randn(1, 4, s, s, generator=g) for n, s in [("a", 8), ("b", 4)]}
    student = {n: torch.randn_like(f) for n, f in teacher.items()}
    expected = F.interpolate(
        layer_distance(teacher["a"], student["a"]), size=(16, 16), mode="bilinear"
    ) * F.interpolate(layer_distance(teacher["b"], student["b"]), size=(16, 16), mode="bilinear")
    torch.testing.assert_close(stfpm_anomaly_map(teacher, student, (16, 16)), expected)


def test_one_perfect_layer_zeroes_the_whole_map() -> None:
    # A product: if any layer sees no difference at a place, the map is 0 there.
    t = {"a": torch.randn(1, 4, 8, 8), "b": torch.randn(1, 4, 4, 4)}
    s = {"a": t["a"].clone(), "b": torch.randn(1, 4, 4, 4)}
    assert stfpm_anomaly_map(t, s, (16, 16)).abs().max() < 1e-6
