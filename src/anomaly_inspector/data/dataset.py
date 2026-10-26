import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from anomaly_inspector.preprocessing import preprocess


class ManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    """Images from manifest rows, preprocessed exactly as the API will preprocess them.

    Pass the rows to use, for example one category and one split:
    `manifest[(manifest.category == "bottle") & (manifest.split == "train")]`.
    """

    def __init__(self, rows: pd.DataFrame, size: int) -> None:
        # Our own copy, so later edits to the caller's DataFrame cannot leak in.
        self.rows = rows.reset_index(drop=True)
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[i]
        with Image.open(str(row["path"])) as image:
            return torch.from_numpy(preprocess(image, self.size)), int(row["label"])
