"""
Opportunity Activity Recognition dataset loader.

Long continuous sensor recordings at 30 Hz for temporal activity segmentation.
This is the benchmark where episodic memory and temporal context should matter —
past activity context influences current classification.

Official Opportunity baseline setup:
- 113 body-worn sensor channels (7 IMUs + accelerometers)
- Locomotion task: 4 classes (Stand, Walk, Sit, Lie) + Null
- Per-timestep classification (temporal segmentation, not windowed)
- Train: ADL1-3 and Drill for Subjects 1-4
- Test: ADL4-5 for Subjects 1-4

The key experiment: vary sequence length (128, 512, 2048, 8192) and measure
whether episodic memory helps at longer horizons.

Reference: Roggen et al., "Collecting complex activity datasets in highly
rich networked sensor environments", INSS 2010.
"""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset

OPPORTUNITY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"
)

# The raw file begins with time, followed by 133 body-worn sensor channels.
# The official challenge representation removes 20 quaternion channels from
# five body IMUs, leaving 113 sensor channels.
_BODY_SENSOR_RAW_COLS = range(1, 134)
_QUATERNION_COLS = {
    *range(46, 50),
    *range(59, 63),
    *range(72, 76),
    *range(85, 89),
    *range(98, 102),
}
BODY_SENSOR_COLS = [column for column in _BODY_SENSOR_RAW_COLS if column not in _QUATERNION_COLS]

# Locomotion label column (0-indexed): column 244 in 1-indexed = index 243
LOCOMOTION_LABEL_COL = 243  # 0-indexed

LOCOMOTION_CLASSES = {
    1: "Stand",
    2: "Walk",
    4: "Sit",
    5: "Lie",
}
NUM_LOCOMOTION_CLASSES = 4  # excluding Null (0)

# Standard train/test split files
TRAIN_FILES = [
    filename
    for subject in range(1, 5)
    for filename in (
        f"S{subject}-ADL1.dat",
        f"S{subject}-ADL2.dat",
        f"S{subject}-ADL3.dat",
        f"S{subject}-Drill.dat",
    )
]

TEST_FILES = [f"S{subject}-ADL{run}.dat" for subject in range(1, 5) for run in (4, 5)]


def download_opportunity(data_dir: str | Path) -> Path:
    """Download and extract Opportunity dataset if not present."""
    data_dir = Path(data_dir)
    dataset_dir = data_dir / "OpportunityUCIDataset"
    zip_path = data_dir / "OpportunityUCIDataset.zip"

    if dataset_dir.exists():
        return dataset_dir / "dataset"

    data_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Downloading Opportunity dataset to {zip_path}...")
        urlretrieve(OPPORTUNITY_URL, zip_path)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Download complete ({size_mb:.1f} MB).")

    print(f"Extracting to {data_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print("Extraction complete.")

    return dataset_dir / "dataset"


def _load_dat_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single .dat file and extract sensor channels + locomotion labels.

    Returns:
        sensors: (num_timesteps, num_channels) float32
        labels: (num_timesteps,) int — locomotion class (0=Null, remapped 1-4)
    """
    # Each .dat file is space-separated, 250 columns
    raw = np.loadtxt(filepath)

    # Extract body sensor columns (0-indexed)
    sensors = raw[:, BODY_SENSOR_COLS].astype(np.float32)

    # Extract locomotion label (0-indexed)
    loco_raw = raw[:, LOCOMOTION_LABEL_COL]

    # Handle NaN labels (missing annotations) — treat as Null (0)
    loco_raw = np.nan_to_num(loco_raw, nan=0.0)
    loco_labels = loco_raw.astype(int)

    # Remap: original {1:Stand, 2:Walk, 4:Sit, 5:Lie} → {1,2,3,4}, everything else=0 (Null)
    label_map = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4}
    labels = np.array(
        [label_map.get(label, 0) for label in loco_labels],
        dtype=int,
    )

    # Interpolate missing values channel by channel. Fully missing channels
    # fall back to zero.
    row_indices = np.arange(sensors.shape[0])
    for channel in range(sensors.shape[1]):
        valid = np.isfinite(sensors[:, channel])
        if valid.any():
            sensors[:, channel] = np.interp(
                row_indices,
                row_indices[valid],
                sensors[valid, channel],
            )
        else:
            sensors[:, channel] = 0.0

    return sensors, labels


def _load_split(dataset_dir: Path, file_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate all files in a split."""
    all_sensors = []
    all_labels = []

    for fname in file_list:
        filepath = dataset_dir / fname
        if not filepath.exists():
            print(f"  Warning: {filepath} not found, skipping")
            continue
        sensors, labels = _load_dat_file(filepath)
        all_sensors.append(sensors)
        all_labels.append(labels)
        print(f"  Loaded {fname}: {len(labels)} timesteps")

    return np.concatenate(all_sensors, axis=0), np.concatenate(all_labels, axis=0)


class OpportunityDataset(Dataset):
    """
    PyTorch Dataset for Opportunity locomotion recognition.

    Segments the continuous sensor stream into windows of configurable length.
    Per-timestep labels enable temporal segmentation evaluation.

    Args:
        data_dir: Directory to download/store the dataset.
        split: "train" or "test".
        window_size: Number of timesteps per sample.
        stride: Stride between consecutive windows.
        normalize: Standardize using training set statistics.
        include_null: If False, discard windows that are majority Null class.
    """

    SAMPLE_RATE_HZ = 30.0

    def __init__(
        self,
        data_dir: str | Path = "./data",
        split: str = "train",
        window_size: int = 128,
        stride: int | None = None,
        normalize: bool = True,
        include_null: bool = False,
    ):
        assert split in ("train", "test")

        dataset_dir = download_opportunity(data_dir)

        file_list = TRAIN_FILES if split == "train" else TEST_FILES
        print(f"\nLoading Opportunity {split} split...")
        sensors, labels = _load_split(dataset_dir, file_list)

        self.num_channels = sensors.shape[1]
        self.window_size = window_size
        self.split = split

        if stride is None:
            stride = window_size // 2

        # Normalize
        if normalize:
            if split == "train":
                self._mean = sensors.mean(axis=0)
                self._std = sensors.std(axis=0)
                self._std[self._std < 1e-8] = 1.0
            else:
                # Load train stats
                train_sensors, _ = _load_split(dataset_dir, TRAIN_FILES)
                self._mean = train_sensors.mean(axis=0)
                self._std = train_sensors.std(axis=0)
                self._std[self._std < 1e-8] = 1.0
                del train_sensors

            sensors = (sensors - self._mean) / self._std
        else:
            self._mean = np.zeros(self.num_channels)
            self._std = np.ones(self.num_channels)

        # Segment into windows
        self.windows = []
        self.window_labels = []

        for start in range(0, len(sensors) - window_size + 1, stride):
            end = start + window_size
            win_labels = labels[start:end]

            # Skip windows that are majority Null (class 0)
            if not include_null:
                non_null_ratio = (win_labels > 0).mean()
                if non_null_ratio < 0.5:
                    continue

            self.windows.append(sensors[start:end])
            self.window_labels.append(win_labels)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.window_labels = np.array(self.window_labels, dtype=np.int64)

        # For classification: majority vote label per window
        # For segmentation: per-timestep labels
        self.majority_labels = np.array(
            [
                np.bincount(wl[wl > 0], minlength=NUM_LOCOMOTION_CLASSES + 1).argmax()
                if (wl > 0).any()
                else 0
                for wl in self.window_labels
            ],
            dtype=np.int64,
        )

        print(f"  {split}: {len(self)} windows (size={window_size}, stride={stride})")
        print(f"  Channels: {self.num_channels}")

        # Class distribution
        if len(self) > 0:
            unique, counts = np.unique(self.majority_labels, return_counts=True)
            for u, c in zip(unique, counts):
                name = {0: "Null", 1: "Stand", 2: "Walk", 3: "Sit", 4: "Lie"}.get(u, f"?{u}")
                print(f"    {name}: {c} ({c / len(self) * 100:.1f}%)")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
                'sensors': (window_size, num_channels) float32
                'labels_per_step': (window_size,) int64 — per-timestep labels
                'label': int — majority class for the window
        """
        return {
            "sensors": torch.from_numpy(self.windows[idx]),
            "labels_per_step": torch.from_numpy(self.window_labels[idx]),
            "label": int(self.majority_labels[idx]),
        }

    def get_config_kwargs(self) -> dict:
        """Return kwargs for ArcMindConfig matched to this dataset."""
        return {
            "num_sensor_channels": self.num_channels,
            "sensor_freq_hz": self.SAMPLE_RATE_HZ,
            "action_dim": NUM_LOCOMOTION_CLASSES + 1,  # include Null as class 0
        }

    def __repr__(self) -> str:
        return (
            f"OpportunityDataset(split='{self.split}', "
            f"windows={len(self)}, "
            f"window_size={self.window_size}, "
            f"channels={self.num_channels})"
        )
