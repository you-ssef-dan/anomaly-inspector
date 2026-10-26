import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from anomaly_inspector.config import StfpmConfig
from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.models.stfpm import STFPM, stfpm_loss

EpochCallback = Callable[[int, dict[str, float]], None]


@dataclass(frozen=True)
class TrainResult:
    best_val_loss: float
    best_epoch: int  # counted from 0
    epochs_run: int
    history: list[dict[str, float]]  # one entry per epoch: train_loss, val_loss, seconds


def stack_images(dataset: ManifestDataset) -> torch.Tensor:
    """All images of a dataset as one (N, 3, H, W) tensor, decoded once instead of every epoch."""
    return torch.stack([dataset[i][0] for i in range(len(dataset))])


def mean_loss(model: STFPM, images: torch.Tensor, batch_size: int) -> float:
    """Average loss per image in eval mode, so a smaller last batch does not weigh more."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in images.split(batch_size):
            total += stfpm_loss(*model(batch)).item() * len(batch)
    return total / len(images)


def train_stfpm(
    model: STFPM,
    train_images: torch.Tensor,
    val_images: torch.Tensor,
    config: StfpmConfig,
    checkpoint_path: Path,
    on_epoch: EpochCallback | None = None,
) -> TrainResult:
    """Train the student on good images; stop when the val_normal loss stops improving.

    Each better epoch is saved to checkpoint_path. When training ends, the model holds the best
    student, not the last one, and is left in eval mode.
    """
    generator = torch.Generator().manual_seed(config.seed)  # shuffling order is reproducible
    optimizer = torch.optim.SGD(
        model.student.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, waited = float("inf"), -1, 0
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        start = time.perf_counter()
        model.train()
        total = 0.0
        for indices in torch.randperm(len(train_images), generator=generator).split(
            config.batch_size
        ):
            loss = stfpm_loss(*model(train_images[indices]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        metrics = {
            "train_loss": total / len(train_images),
            "val_loss": mean_loss(model, val_images, config.batch_size),
            "seconds": time.perf_counter() - start,
        }
        history.append(metrics)
        if metrics["val_loss"] < best - config.min_delta:
            best, best_epoch, waited = metrics["val_loss"], epoch, 0
            torch.save(model.student.state_dict(), checkpoint_path)
        else:
            waited += 1
        if on_epoch is not None:  # after the checkpoint, so a callback cannot alter what is saved
            on_epoch(epoch, metrics)
        if waited >= config.patience:
            break
    model.student.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    return TrainResult(best, best_epoch, len(history), history)
