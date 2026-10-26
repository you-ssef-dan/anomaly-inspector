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


class Params(Strict):
    data: DataConfig


def load_params(path: Path = Path("params.yaml")) -> Params:
    return Params.model_validate(yaml.safe_load(path.read_text()))
