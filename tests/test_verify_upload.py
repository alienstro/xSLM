import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_upload import REQUIRED, check_files  # noqa: E402


def complete():
    return [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "README.md",
        "gguf/xslm-70m-BF16.gguf",
        "gguf/xslm-70m-Q8_0.gguf",
        "gguf/xslm-70m-Q6_K.gguf",
        "gguf/xslm-70m-Q4_K_M.gguf",
    ]


def test_a_complete_repository_passes():
    assert check_files(complete(), "xslm-70m") == []


def test_a_missing_quantization_is_reported():
    files = [f for f in complete() if "Q6_K" not in f]
    assert check_files(files, "xslm-70m") == ["gguf/xslm-70m-Q6_K.gguf"]


def test_a_missing_weight_file_is_reported():
    files = [f for f in complete() if f != "model.safetensors"]
    assert "model.safetensors" in check_files(files, "xslm-70m")


def test_an_empty_repository_reports_every_file():
    assert len(check_files([], "xslm-70m")) == len(REQUIRED)


def test_the_required_list_names_all_four_gguf_files():
    names = " ".join(REQUIRED)
    for quant in ("BF16", "Q8_0", "Q6_K", "Q4_K_M"):
        assert quant in names
