import numpy as np
import pytest
import torch

from xslm.data import PackedDataset


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(5000, dtype=np.uint16).tofile(path)
    return path


def test_a_sample_holds_the_requested_length(token_file):
    sample = PackedDataset(token_file, seq_len=128)[0]
    assert sample["input_ids"].shape == (128,)
    assert sample["input_ids"].dtype == torch.long


def test_the_labels_equal_the_inputs(token_file):
    sample = PackedDataset(token_file, seq_len=64)[3]
    assert torch.equal(sample["input_ids"], sample["labels"])


def test_no_sample_reads_past_the_end_of_the_file(token_file):
    dataset = PackedDataset(token_file, seq_len=64, seed=7)
    for index in range(len(dataset)):
        values = dataset[index]["input_ids"]
        assert values.max().item() < 5000
        assert values.shape == (64,)


def test_the_same_index_gives_the_same_sample(token_file):
    dataset = PackedDataset(token_file, seq_len=32, seed=11)
    assert torch.equal(dataset[5]["input_ids"], dataset[5]["input_ids"])


def test_a_file_shorter_than_one_sequence_raises(tmp_path):
    path = tmp_path / "short.bin"
    np.arange(10, dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError, match="shorter than"):
        PackedDataset(path, seq_len=64)
