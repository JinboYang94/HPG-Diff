"""Shared access to the HPG-Diff training and benchmark datasets."""
from functools import lru_cache

from datasets import load_dataset


DATASET_REPO = "scottyjb/HPGDiff"


@lru_cache(maxsize=3)
def _load_split(config, split):
    return load_dataset(DATASET_REPO, config, split=split)


def load_training_data():
    return _load_split("train", "train")


def load_test_data(test_level):
    level = str(test_level)
    if level not in {"1", "2"}:
        raise ValueError("test_level must be 1 or 2.")
    return _load_split("test", f"test_level_{level}")
