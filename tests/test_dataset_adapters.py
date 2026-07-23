"""Unit tests for dataset adapters without external downloads."""

from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from arcmind.data import antmaze, mujoco_locomotion, opportunity, uci_har


def antmaze_episode(length: int = 3) -> SimpleNamespace:
    observations = {
        "observation": np.arange((length + 1) * 27, dtype=np.float32).reshape(
            length + 1,
            27,
        ),
        "achieved_goal": np.ones((length + 1, 2), dtype=np.float32),
        "desired_goal": np.full((length + 1, 2), 2.0, dtype=np.float32),
    }
    return SimpleNamespace(
        observations=observations,
        actions=np.arange(length * 8, dtype=np.float32).reshape(length, 8),
        rewards=np.arange(length, dtype=np.float32),
    )


def mujoco_episode(
    observation_dim: int = 17,
    action_dim: int = 6,
    length: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        observations=np.arange(
            (length + 1) * observation_dim,
            dtype=np.float32,
        ).reshape(length + 1, observation_dim),
        actions=np.arange(length * action_dim, dtype=np.float32).reshape(
            length,
            action_dim,
        ),
        rewards=np.arange(length, dtype=np.float32),
    )


def fake_minari(episode: SimpleNamespace) -> SimpleNamespace:
    dataset = SimpleNamespace(iterate_episodes=lambda: iter([episode]))
    return SimpleNamespace(load_dataset=lambda *_args, **_kwargs: dataset)


class TestAntMazeDataset:
    def test_flatten_observation(self):
        episode = antmaze_episode()
        flattened = antmaze._flatten_obs(episode.observations)
        assert flattened.shape == (4, 31)
        assert flattened.dtype == np.float32

    def test_requires_minari(self, monkeypatch):
        monkeypatch.setattr(antmaze, "minari", None)
        with pytest.raises(ImportError, match="minari is required"):
            antmaze.AntMazeDataset()

    def test_rejects_unknown_variant(self, monkeypatch):
        monkeypatch.setattr(antmaze, "minari", fake_minari(antmaze_episode()))
        with pytest.raises(AssertionError, match="variant must be"):
            antmaze.AntMazeDataset(variant="unknown")

    @pytest.mark.parametrize("normalize", [False, True])
    def test_episode_padding_and_metadata(self, monkeypatch, normalize):
        monkeypatch.setattr(antmaze, "minari", fake_minari(antmaze_episode()))
        dataset = antmaze.AntMazeDataset(
            max_episode_len=5,
            normalize=normalize,
        )

        sample = dataset[0]
        assert len(dataset) == 1
        assert sample["observations"].shape == (5, 31)
        assert sample["actions"].shape == (5, 8)
        assert sample["rewards"].shape == (5,)
        assert sample["mask"].tolist() == [True, True, True, False, False]
        assert sample["length"] == 3
        assert dataset.get_config_kwargs() == {
            "num_sensor_channels": 31,
            "action_dim": 8,
        }
        assert dataset.get_normalization_stats()["obs_mean"].shape == (31,)
        assert "episodes=1" in repr(dataset)


class TestMuJoCoDataset:
    def test_requires_minari(self, monkeypatch):
        monkeypatch.setattr(mujoco_locomotion, "minari", None)
        with pytest.raises(ImportError, match="minari is required"):
            mujoco_locomotion.MuJoCoLocomotionDataset()

    def test_rejects_unknown_task(self, monkeypatch):
        monkeypatch.setattr(
            mujoco_locomotion,
            "minari",
            fake_minari(mujoco_episode()),
        )
        with pytest.raises(AssertionError, match="Unknown task"):
            mujoco_locomotion.MuJoCoLocomotionDataset(task="ant")

    @pytest.mark.parametrize("normalize", [False, True])
    def test_episode_padding_and_metadata(self, monkeypatch, normalize):
        monkeypatch.setattr(
            mujoco_locomotion,
            "minari",
            fake_minari(mujoco_episode()),
        )
        dataset = mujoco_locomotion.MuJoCoLocomotionDataset(
            max_episode_len=3,
            normalize=normalize,
        )

        sample = dataset[0]
        assert len(dataset) == 1
        assert sample["observations"].shape == (3, 17)
        assert sample["actions"].shape == (3, 6)
        assert sample["mask"].all()
        assert sample["length"] == 3
        assert dataset.get_config_kwargs() == {
            "num_sensor_channels": 17,
            "action_dim": 6,
        }
        assert dataset.get_normalization_stats()["obs_mean"].shape == (17,)
        assert dataset.get_env_name() == "HalfCheetah-v4"
        assert "halfcheetah" in repr(dataset)


class TestOpportunityDataset:
    def test_official_channel_and_split_contract(self):
        assert len(opportunity.BODY_SENSOR_COLS) == 113
        assert len(opportunity.TRAIN_FILES) == 16
        assert len(opportunity.TEST_FILES) == 8
        assert "S4-Drill.dat" in opportunity.TRAIN_FILES
        assert "S4-ADL5.dat" in opportunity.TEST_FILES

    def test_raw_file_mapping_and_interpolation(self, monkeypatch):
        raw = np.zeros((3, 250), dtype=np.float64)
        raw[:, opportunity.BODY_SENSOR_COLS] = np.arange(3)[:, None]
        raw[1, opportunity.BODY_SENSOR_COLS[0]] = np.nan
        raw[:, opportunity.BODY_SENSOR_COLS[1]] = np.nan
        raw[:, opportunity.LOCOMOTION_LABEL_COL] = [1, 4, np.nan]
        monkeypatch.setattr(opportunity.np, "loadtxt", lambda _path: raw)

        sensors, labels = opportunity._load_dat_file("unused.dat")

        assert sensors.shape == (3, 113)
        assert sensors.dtype == np.float32
        assert sensors[:, 0].tolist() == [0.0, 1.0, 2.0]
        assert sensors[:, 1].tolist() == [0.0, 0.0, 0.0]
        assert labels.tolist() == [1, 3, 0]

    def test_split_skips_missing_files(self, monkeypatch, tmp_path):
        present = tmp_path / "present.dat"
        present.touch()
        monkeypatch.setattr(
            opportunity,
            "_load_dat_file",
            lambda _path: (
                np.ones((2, 113), dtype=np.float32),
                np.ones(2, dtype=np.int64),
            ),
        )

        sensors, labels = opportunity._load_split(
            tmp_path,
            ["missing.dat", "present.dat"],
        )

        assert sensors.shape == (2, 113)
        assert labels.tolist() == [1, 1]

    def test_download_extracts_existing_archive(self, tmp_path):
        archive = tmp_path / "OpportunityUCIDataset.zip"
        with ZipFile(archive, "w") as handle:
            handle.writestr(
                "OpportunityUCIDataset/dataset/example.dat",
                "example",
            )

        extracted = opportunity.download_opportunity(tmp_path)
        assert extracted == tmp_path / "OpportunityUCIDataset" / "dataset"
        assert (extracted / "example.dat").is_file()
        assert opportunity.download_opportunity(tmp_path) == extracted

    @pytest.mark.parametrize(
        ("split", "normalize", "include_null", "expected_windows"),
        [
            ("train", True, True, 5),
            ("test", True, True, 5),
            ("train", False, False, 4),
        ],
    )
    def test_windowing_and_metadata(
        self,
        monkeypatch,
        tmp_path,
        split,
        normalize,
        include_null,
        expected_windows,
    ):
        sensors = np.arange(12 * 113, dtype=np.float32).reshape(12, 113)
        labels = np.array([0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 3, 3])
        monkeypatch.setattr(
            opportunity,
            "download_opportunity",
            lambda _path: tmp_path,
        )
        monkeypatch.setattr(
            opportunity,
            "_load_split",
            lambda _path, _files: (sensors.copy(), labels.copy()),
        )

        dataset = opportunity.OpportunityDataset(
            data_dir=tmp_path,
            split=split,
            window_size=4,
            stride=2,
            normalize=normalize,
            include_null=include_null,
        )

        assert len(dataset) == expected_windows
        sample = dataset[0]
        assert sample["sensors"].shape == (4, 113)
        assert sample["labels_per_step"].shape == (4,)
        assert isinstance(sample["label"], int)
        assert dataset.get_config_kwargs() == {
            "num_sensor_channels": 113,
            "sensor_freq_hz": 30.0,
            "action_dim": 5,
        }
        assert f"split='{split}'" in repr(dataset)


class TestUCIHARDatasetUnit:
    def test_download_extracts_existing_archive(self, tmp_path):
        archive = tmp_path / "UCI_HAR_Dataset.zip"
        with ZipFile(archive, "w") as handle:
            handle.writestr("UCI HAR Dataset/train/example.txt", "example")

        extracted = uci_har.download_uci_har(tmp_path)
        assert extracted == tmp_path / "UCI HAR Dataset"
        assert (extracted / "train" / "example.txt").is_file()
        assert uci_har.download_uci_har(tmp_path) == extracted

    def test_signal_and_label_loading(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            uci_har.np,
            "loadtxt",
            lambda path, **_kwargs: (
                np.array([1, 6]) if path.name == "y_train.txt" else np.ones((2, 128))
            ),
        )

        signals = uci_har._load_signals(tmp_path, "train")
        labels = uci_har._load_labels(tmp_path, "train")

        assert signals.shape == (2, 128, 6)
        assert signals.dtype == np.float32
        assert labels.tolist() == [0, 5]

    @pytest.mark.parametrize(
        ("split", "normalize"), [("train", True), ("test", True), ("train", False)]
    )
    def test_dataset_contract(self, monkeypatch, tmp_path, split, normalize):
        signals = np.arange(3 * 128 * 6, dtype=np.float32).reshape(3, 128, 6)
        labels = np.array([0, 1, 2])
        monkeypatch.setattr(uci_har, "download_uci_har", lambda _path: tmp_path)
        monkeypatch.setattr(
            uci_har,
            "_load_signals",
            lambda _path, _split: signals.copy(),
        )
        monkeypatch.setattr(
            uci_har,
            "_load_labels",
            lambda _path, _split: labels.copy(),
        )

        dataset = uci_har.UCIHARDataset(
            data_dir=tmp_path,
            split=split,
            normalize=normalize,
        )

        sensor_window, label = dataset[1]
        assert len(dataset) == 3
        assert sensor_window.shape == (128, 6)
        assert sensor_window.dtype == torch.float32
        assert label == 1
        assert dataset.get_config_kwargs() == {
            "num_sensor_channels": 6,
            "sensor_freq_hz": 50.0,
            "action_dim": 6,
        }
        assert f"split='{split}'" in repr(dataset)
