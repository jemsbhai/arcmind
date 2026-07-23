"""
UCI HAR Dataset loader.

Downloads and loads the UCI Human Activity Recognition dataset, using the
raw inertial signals (not pre-computed features) to validate sensor-native
tokenization.

Dataset: 30 subjects wearing a smartphone (Samsung Galaxy S II) on the waist.
Sensors: 3-axis accelerometer + 3-axis gyroscope at 50 Hz.
Activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
Segmentation: 128-sample windows with 50% overlap (2.56 seconds per window).
Split: 21 subjects for train (7,352 windows), 9 subjects for test (2,947 windows).

Reference: Anguita et al., "A Public Domain Dataset for Human Activity Recognition
Using Smartphones", ESANN 2013.
"""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset

UCI_HAR_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
)

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# Raw inertial signal files (6 channels: 3 accel + 3 gyro)
SIGNAL_FILES = [
    "body_acc_x_{}.txt",
    "body_acc_y_{}.txt",
    "body_acc_z_{}.txt",
    "body_gyro_x_{}.txt",
    "body_gyro_y_{}.txt",
    "body_gyro_z_{}.txt",
]


def download_uci_har(data_dir: str | Path) -> Path:
    """
    Download and extract UCI HAR dataset if not already present.

    Args:
        data_dir: Directory to store the dataset.

    Returns:
        Path to the extracted dataset root (contains train/ and test/).
    """
    data_dir = Path(data_dir)
    dataset_dir = data_dir / "UCI HAR Dataset"
    zip_path = data_dir / "UCI_HAR_Dataset.zip"

    if dataset_dir.exists():
        return dataset_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Downloading UCI HAR Dataset to {zip_path}...")
        urlretrieve(UCI_HAR_URL, zip_path)
        print("Download complete.")

    print(f"Extracting to {data_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print("Extraction complete.")

    return dataset_dir


def _load_signals(dataset_dir: Path, split: str) -> np.ndarray:
    """
    Load raw inertial signals for a given split.

    Args:
        dataset_dir: Path to "UCI HAR Dataset" root.
        split: "train" or "test".

    Returns:
        Array of shape (num_windows, 128, 6) — 6 sensor channels.
    """
    signals = []
    signal_dir = dataset_dir / split / "Inertial Signals"

    for signal_file in SIGNAL_FILES:
        filename = signal_file.format(split)
        filepath = signal_dir / filename
        # Each file: num_windows rows, 128 space-separated values per row
        data = np.loadtxt(filepath)
        signals.append(data)

    # Stack: (6, num_windows, 128) -> (num_windows, 128, 6)
    signals = np.stack(signals, axis=-1)
    return signals.astype(np.float32)


def _load_labels(dataset_dir: Path, split: str) -> np.ndarray:
    """Load activity labels (1-indexed) for a given split."""
    filepath = dataset_dir / split / f"y_{split}.txt"
    labels = np.loadtxt(filepath, dtype=int)
    # Convert to 0-indexed
    return labels - 1


class UCIHARDataset(Dataset):
    """
    PyTorch Dataset for UCI HAR raw inertial signals.

    Each sample is a (sensor_window, label) pair where:
    - sensor_window: shape (128, 6) — 128 timesteps x 6 channels
    - label: int in [0, 5] — activity class

    Args:
        data_dir: Directory to download/store the dataset.
        split: "train" or "test".
        normalize: If True, standardize each channel to zero mean, unit variance
                   using training set statistics.
    """

    NUM_CHANNELS = 6
    SEQ_LEN = 128
    NUM_CLASSES = 6
    SAMPLE_RATE_HZ = 50.0

    def __init__(
        self,
        data_dir: str | Path = "./data",
        split: str = "train",
        normalize: bool = True,
    ):
        assert split in ("train", "test"), f"split must be 'train' or 'test', got '{split}'"

        dataset_dir = download_uci_har(data_dir)

        self.signals = _load_signals(dataset_dir, split)
        self.labels = _load_labels(dataset_dir, split)
        self.split = split

        if normalize:
            if split == "train":
                # Compute stats from training data
                self._mean = self.signals.mean(axis=(0, 1))  # (6,)
                self._std = self.signals.std(axis=(0, 1))  # (6,)
                self._std[self._std < 1e-8] = 1.0  # prevent division by zero
            else:
                # For test set, load training data to compute stats
                train_signals = _load_signals(dataset_dir, "train")
                self._mean = train_signals.mean(axis=(0, 1))
                self._std = train_signals.std(axis=(0, 1))
                self._std[self._std < 1e-8] = 1.0

            self.signals = (self.signals - self._mean) / self._std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Returns:
            sensor_window: shape (128, 6), float32 tensor.
            label: int, activity class [0, 5].
        """
        sensor_window = torch.from_numpy(self.signals[idx])
        label = int(self.labels[idx])
        return sensor_window, label

    def get_config_kwargs(self) -> dict:
        """
        Return kwargs to construct an ArcMindConfig matched to this dataset.

        Usage:
            dataset = UCIHARDataset(split="train")
            config = ArcMindConfig(**dataset.get_config_kwargs())
        """
        return {
            "num_sensor_channels": self.NUM_CHANNELS,
            "sensor_freq_hz": self.SAMPLE_RATE_HZ,
            "action_dim": self.NUM_CLASSES,
        }

    def __repr__(self) -> str:
        return (
            f"UCIHARDataset(split='{self.split}', "
            f"samples={len(self)}, "
            f"channels={self.NUM_CHANNELS}, "
            f"seq_len={self.SEQ_LEN}, "
            f"classes={self.NUM_CLASSES})"
        )
