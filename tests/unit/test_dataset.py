from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.data.manifest import build_manifest
from anomaly_inspector.preprocessing import preprocess

SIZE = 32


@pytest.fixture
def manifest(fake_mvtec: Path) -> pd.DataFrame:
    return build_manifest(fake_mvtec, ["bottle", "screw"])


def test_length_is_the_number_of_rows(manifest: pd.DataFrame) -> None:
    assert len(ManifestDataset(manifest, SIZE)) == len(manifest)


def test_item_is_a_float_tensor_and_an_int_label(manifest: pd.DataFrame) -> None:
    image, label = ManifestDataset(manifest, SIZE)[0]
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    assert image.shape == (3, SIZE, SIZE)
    assert isinstance(label, int)


def test_image_is_exactly_what_preprocess_gives(manifest: pd.DataFrame) -> None:
    # The whole point: training sees the same numbers the API will compute.
    dataset = ManifestDataset(manifest, SIZE)
    for i in range(len(dataset)):
        image, _ = dataset[i]
        with Image.open(str(manifest.iloc[i]["path"])) as original:
            np.testing.assert_array_equal(image.numpy(), preprocess(original, SIZE))


def test_labels_follow_the_manifest(manifest: pd.DataFrame) -> None:
    dataset = ManifestDataset(manifest, SIZE)
    assert [dataset[i][1] for i in range(len(dataset))] == list(manifest["label"])


def test_item_i_is_the_ith_of_the_given_rows(manifest: pd.DataFrame) -> None:
    # Shuffled rows: the index is scrambled and nothing is in manifest order any more.
    shuffled = manifest.sample(frac=1, random_state=0)
    dataset = ManifestDataset(shuffled, SIZE)
    for i in range(len(dataset)):
        image, label = dataset[i]
        assert label == shuffled.iloc[i]["label"]
        with Image.open(str(shuffled.iloc[i]["path"])) as original:
            np.testing.assert_array_equal(image.numpy(), preprocess(original, SIZE))


def test_grayscale_images_come_out_with_three_channels(manifest: pd.DataFrame) -> None:
    screws = manifest[manifest["mode"] == "L"]
    assert len(screws) > 0
    image, _ = ManifestDataset(screws, SIZE)[0]
    assert image.shape == (3, SIZE, SIZE)


def test_the_input_rows_are_not_modified(manifest: pd.DataFrame) -> None:
    subset = manifest[manifest["category"] == "screw"]
    before = subset.copy()
    ManifestDataset(subset, SIZE)
    pd.testing.assert_frame_equal(subset, before)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_dataloader_batches_it(manifest: pd.DataFrame, num_workers: int) -> None:
    # forkserver, not fork: forking a process that runs torch threads can deadlock, and the
    # workers then receive the dataset pickled, which proves it can be sent to them.
    context = "forkserver" if num_workers else None
    loader = DataLoader(
        ManifestDataset(manifest, SIZE),
        batch_size=4,
        num_workers=num_workers,
        multiprocessing_context=context,
    )
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, SIZE, SIZE)
    assert labels.dtype == torch.int64
    assert sum(len(batch[1]) for batch in loader) == len(manifest)
