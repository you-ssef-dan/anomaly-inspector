import copy
from pathlib import Path
from typing import Any

import pytest
import torch

from anomaly_inspector.config import StfpmConfig
from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.data.manifest import build_manifest
from anomaly_inspector.models.stfpm import STFPM, stfpm_loss
from anomaly_inspector.training.stfpm import TrainResult, mean_loss, stack_images, train_stfpm

SIZE = 64


def tiny_model(seed: int = 0) -> STFPM:
    torch.manual_seed(seed)
    return STFPM.from_backbone("resnet18", pretrained=False)


def good_images(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ramp = torch.linspace(-1, 1, SIZE)
    base = (ramp[None, :] + ramp[:, None]).expand(3, SIZE, SIZE) * 0.5
    return torch.stack([base + 0.05 * torch.randn(3, SIZE, SIZE, generator=g) for _ in range(n)])


TRAIN, VAL = good_images(8, seed=0), good_images(5, seed=1)


def config(**overrides: Any) -> StfpmConfig:
    values: dict[str, Any] = {"epochs": 8, "lr": 0.4, "batch_size": 4, "patience": 8, "seed": 0}
    return StfpmConfig(**(values | overrides))


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One short training run shared by the tests below (about 2 seconds)."""
    model = tiny_model()
    teacher_before = copy.deepcopy(model.teacher.state_dict())
    checkpoint = tmp_path_factory.mktemp("run") / "nested" / "student.pt"
    seen: list[tuple[int, dict[str, float]]] = []
    result = train_stfpm(
        model, TRAIN, VAL, config(), checkpoint, on_epoch=lambda e, m: seen.append((e, m))
    )
    return {
        "model": model,
        "result": result,
        "checkpoint": checkpoint,
        "teacher_before": teacher_before,
        "seen": seen,
    }


def val_losses(result: TrainResult) -> list[float]:
    return [epoch["val_loss"] for epoch in result.history]


def test_history_has_one_entry_per_epoch(run: dict[str, Any]) -> None:
    result = run["result"]
    assert result.epochs_run == len(result.history) == 8
    assert all(set(epoch) == {"train_loss", "val_loss", "seconds"} for epoch in result.history)


def test_training_lowers_the_loss(run: dict[str, Any]) -> None:
    history = run["result"].history
    assert history[-1]["train_loss"] < 0.5 * history[0]["train_loss"]


def test_best_epoch_is_the_one_with_the_lowest_validation_loss(run: dict[str, Any]) -> None:
    result = run["result"]
    losses = val_losses(result)
    assert result.best_val_loss == min(losses)
    assert result.best_epoch == losses.index(min(losses))


def test_the_checkpoint_reloads_to_the_best_validation_loss(run: dict[str, Any]) -> None:
    fresh = tiny_model()  # same seed, so the same teacher; the student gets the saved weights
    fresh.student.load_state_dict(torch.load(run["checkpoint"], weights_only=True))
    assert mean_loss(fresh, VAL, batch_size=4) == pytest.approx(run["result"].best_val_loss)


def test_the_returned_model_holds_the_best_student_in_eval_mode(run: dict[str, Any]) -> None:
    model = run["model"]
    assert not model.student.training
    assert mean_loss(model, VAL, batch_size=4) == pytest.approx(run["result"].best_val_loss)


def test_training_never_changes_the_teacher(run: dict[str, Any]) -> None:
    for name, value in run["model"].teacher.state_dict().items():
        assert torch.equal(value, run["teacher_before"][name])


def test_on_epoch_receives_each_epochs_metrics(run: dict[str, Any]) -> None:
    assert [epoch for epoch, _ in run["seen"]] == list(range(8))
    assert [metrics for _, metrics in run["seen"]] == run["result"].history


def test_early_stopping_after_patience_epochs_without_improvement(tmp_path: Path) -> None:
    # A learning rate this small changes the loss by far less than min_delta: no improvement.
    result = train_stfpm(
        tiny_model(), TRAIN, VAL, config(epochs=20, lr=1e-12, patience=3), tmp_path / "s.pt"
    )
    assert result.best_epoch == 0
    assert result.epochs_run == 1 + 3


def test_same_seed_gives_the_same_run_and_another_seed_a_different_one(tmp_path: Path) -> None:
    def train_losses(seed: int) -> list[float]:
        result = train_stfpm(
            tiny_model(), TRAIN, VAL, config(epochs=3, seed=seed), tmp_path / f"{seed}.pt"
        )
        return [epoch["train_loss"] for epoch in result.history]

    assert train_losses(0) == train_losses(0)
    assert train_losses(0) != train_losses(1)  # different shuffling order


def test_small_improvements_below_min_delta_do_not_count(tmp_path: Path) -> None:
    # The loss really drops here, but never by 10: only the first epoch counts as best.
    result = train_stfpm(
        tiny_model(), TRAIN, VAL, config(epochs=20, min_delta=10.0, patience=3), tmp_path / "s.pt"
    )
    assert val_losses(result)[-1] < val_losses(result)[0]
    assert result.best_epoch == 0
    assert result.epochs_run == 1 + 3


def test_the_best_epoch_wins_even_when_later_epochs_are_worse(tmp_path: Path) -> None:
    # Damage the student right after epoch 2, like a run that suddenly diverges.
    model, checkpoint = tiny_model(), tmp_path / "s.pt"

    def damage_after_epoch_2(epoch: int, _: dict[str, float]) -> None:
        if epoch == 2:
            g = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for p in model.student.parameters():
                    p.add_(5 * torch.randn(p.shape, generator=g))

    result = train_stfpm(
        model, TRAIN, VAL, config(epochs=6), checkpoint, on_epoch=damage_after_epoch_2
    )
    assert result.best_epoch == 2
    assert min(val_losses(result)[3:]) > result.best_val_loss  # later epochs really were worse
    assert mean_loss(model, VAL, batch_size=4) == pytest.approx(result.best_val_loss)
    fresh = tiny_model()
    fresh.student.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert mean_loss(fresh, VAL, batch_size=4) == pytest.approx(result.best_val_loss)


def test_one_epoch_is_plain_sgd_on_seeded_shuffled_batches(tmp_path: Path) -> None:
    # The reference: what one epoch must do, written out step by step.
    settings = config(epochs=1, batch_size=3, seed=5)
    trained = tiny_model()
    train_stfpm(trained, TRAIN, VAL, settings, tmp_path / "s.pt")

    reference = tiny_model()
    optimizer = torch.optim.SGD(
        reference.student.parameters(),
        lr=settings.lr,
        momentum=settings.momentum,
        weight_decay=settings.weight_decay,
    )
    order = torch.randperm(len(TRAIN), generator=torch.Generator().manual_seed(settings.seed))
    reference.train()
    for indices in order.split(settings.batch_size):
        loss = stfpm_loss(*reference(TRAIN[indices]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for name, value in reference.student.state_dict().items():
        torch.testing.assert_close(trained.student.state_dict()[name], value)


def test_the_student_trains_in_train_mode_and_is_validated_in_eval_mode(tmp_path: Path) -> None:
    model, modes = tiny_model(), []
    model.student.register_forward_pre_hook(lambda module, _: modes.append(module.training))
    train_stfpm(model, TRAIN, VAL, config(epochs=1, batch_size=4), tmp_path / "s.pt")
    # 8 training images in batches of 4, then 5 validation images in batches of 4
    assert modes == [True, True, False, False]


def test_mean_loss_weighs_every_image_equally() -> None:
    # Images whose losses differ a lot, in batches of 3 and 2: a plain average of the two batch
    # losses would give the 2 last images as much weight as the 3 first.
    # (Scaling the images would not do: the features are normalized, so scale does not matter.)
    model = tiny_model()
    noise = torch.randn(2, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(9))
    mixed = torch.cat([good_images(3, seed=2), noise])
    first, last = mean_loss(model, mixed[:3], 3), mean_loss(model, mixed[3:], 3)
    assert abs(first - last) > 0.05  # really different
    assert mean_loss(model, mixed, batch_size=3) == pytest.approx(
        mean_loss(model, mixed, batch_size=len(mixed)), rel=1e-5
    )


def test_stack_images_decodes_a_dataset_once(fake_mvtec: Path) -> None:
    manifest = build_manifest(fake_mvtec, ["bottle"])
    dataset = ManifestDataset(manifest, SIZE)
    images = stack_images(dataset)
    assert images.shape == (len(dataset), 3, SIZE, SIZE)
    assert torch.equal(images[2], dataset[2][0])
