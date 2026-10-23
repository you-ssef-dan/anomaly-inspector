import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS = ["train", "val_normal", "calib", "holdout"]


def assign_splits(df: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    """Add a `split` column, category by category.

    train/good is cut into train and val_normal; the test images are halved into calib and
    holdout, stratified by defect type so both halves see every kind of defect.
    """
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("category"):
        train = group[group["source_split"] == "train"]
        test = group[group["source_split"] == "test"]
        fit, val = train_test_split(train, test_size=val_fraction, random_state=seed)
        calib, holdout = train_test_split(
            test, test_size=0.5, stratify=test["defect_type"], random_state=seed
        )
        parts += [
            fit.assign(split="train"),
            val.assign(split="val_normal"),
            calib.assign(split="calib"),
            holdout.assign(split="holdout"),
        ]
    return pd.concat(parts).sort_index().reset_index(drop=True)
