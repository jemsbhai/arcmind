"""Tests for UCI HAR dataset loader."""

import pytest
import torch

from arcmind.data.uci_har import UCIHARDataset

# Mark all tests in this file as requiring network/download
pytestmark = pytest.mark.slow


class TestUCIHARDataset:
    """Tests that require downloading the dataset (run with pytest -m slow)."""

    @pytest.fixture(scope="class")
    @classmethod
    def train_dataset(cls, tmp_path_factory):
        data_dir = tmp_path_factory.mktemp("uci_har")
        return UCIHARDataset(data_dir=data_dir, split="train")

    @pytest.fixture(scope="class")
    @classmethod
    def test_dataset(cls, tmp_path_factory):
        data_dir = tmp_path_factory.mktemp("uci_har_test")
        return UCIHARDataset(data_dir=data_dir, split="test")

    def test_train_length(self, train_dataset):
        assert len(train_dataset) == 7352

    def test_test_length(self, test_dataset):
        assert len(test_dataset) == 2947

    def test_sample_shape(self, train_dataset):
        sensor_window, label = train_dataset[0]
        assert sensor_window.shape == (128, 6)
        assert sensor_window.dtype == torch.float32

    def test_label_range(self, train_dataset):
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            assert 0 <= label <= 5
            break  # just check first to keep it fast

    def test_all_classes_present(self, train_dataset):
        labels = set()
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            labels.add(label)
        assert labels == {0, 1, 2, 3, 4, 5}

    def test_normalization(self, train_dataset):
        """Normalized data should have roughly zero mean per channel."""
        # Check first 100 samples
        signals = torch.stack([train_dataset[i][0] for i in range(100)])
        channel_means = signals.mean(dim=(0, 1))  # (6,)
        # After normalization, means should be close to 0
        assert channel_means.abs().max() < 1.0, (
            f"Channel means after normalization: {channel_means}"
        )

    def test_config_kwargs(self, train_dataset):
        kwargs = train_dataset.get_config_kwargs()
        assert kwargs["num_sensor_channels"] == 6
        assert kwargs["sensor_freq_hz"] == 50.0
        assert kwargs["action_dim"] == 6

    def test_repr(self, train_dataset):
        r = repr(train_dataset)
        assert "train" in r
        assert "7352" in r
