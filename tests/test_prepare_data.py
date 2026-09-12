import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_data import append_tokens, tokens_on_disk  # noqa: E402


def test_an_absent_file_holds_no_tokens(tmp_path):
    assert tokens_on_disk(tmp_path / "missing.bin") == 0


def test_append_returns_the_running_total(tmp_path):
    path = tmp_path / "train.bin"
    assert append_tokens(path, [1, 2, 3]) == 3
    assert append_tokens(path, [4, 5]) == 5
    assert tokens_on_disk(path) == 5


def test_the_file_holds_uint16_values_in_order(tmp_path):
    path = tmp_path / "train.bin"
    append_tokens(path, [7, 8, 9])
    assert np.fromfile(path, dtype=np.uint16).tolist() == [7, 8, 9]


def test_a_token_above_the_uint16_range_raises(tmp_path):
    with pytest.raises(ValueError, match="65535"):
        append_tokens(tmp_path / "train.bin", [70000])
