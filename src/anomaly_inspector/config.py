from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(Strict):
    root: Path
    categories: list[str]
    image_size: int = Field(256, ge=64)
    val_normal_fraction: float = Field(0.1, gt=0, lt=1)
    seed: int = 42


class StfpmConfig(Strict):
    # The STFPM paper (section 4.2): SGD, lr 0.4, batch 32, 100 epochs, best validation loss kept.
    # Momentum and weight decay are common defaults, not values from the paper.
    epochs: int = Field(100, ge=1)
    lr: float = Field(0.4, gt=0)
    batch_size: int = Field(32, ge=1)
    momentum: float = Field(0.9, ge=0, lt=1)
    weight_decay: float = Field(1e-4, ge=0)
    patience: int = Field(10, ge=1)  # epochs without improvement before stopping
    min_delta: float = Field(1e-4, ge=0)  # smaller drops in val loss do not count as improvement
    seed: int = 42


class PatchcoreConfig(Strict):
    # The Phase 2 baseline: WideResNet-50 features, a memory bank of 1% of the patches.
    backbone: str = "wide_resnet50_2"
    coreset_ratio: float = Field(0.01, gt=0, le=1)
    seed: int = 42


class ThresholdConfig(Strict):
    # A missed defect costs 10 times a false reject: an assumption, written in the README.
    miss_cost: float = Field(10, gt=0)
    false_reject_cost: float = Field(1, gt=0)
    # Good-parts-only threshold: the score that 99% of the val_normal parts stay below.
    normal_quantile: float = Field(0.99, gt=0, lt=1)


class Params(Strict):
    data: DataConfig
    patchcore: PatchcoreConfig = PatchcoreConfig()
    threshold: ThresholdConfig = ThresholdConfig()
    stfpm: StfpmConfig = StfpmConfig()


def load_params(path: Path = Path("params.yaml")) -> Params:
    return Params.model_validate(yaml.safe_load(path.read_text()))
