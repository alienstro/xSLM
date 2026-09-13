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
        "gguf/xslm-70m-F16.gguf",
        "gguf/xslm-70m-Q8_0.gguf",
        "gguf/xslm-70m-Q6_K.gguf",
        "gguf/xslm-70m-Q4_K_M.gguf",
        "instruct/config.json",
        "instruct/model.safetensors",
        "instruct/tokenizer.json",
        "instruct/tokenizer_config.json",
        "gguf/xslm-70m-instruct-BF16.gguf",
        "gguf/xslm-70m-instruct-F16.gguf",
        "gguf/xslm-70m-instruct-Q8_0.gguf",
        "gguf/xslm-70m-instruct-Q6_K.gguf",
        "gguf/xslm-70m-instruct-Q4_K_M.gguf",
    ]


def test_a_complete_repository_passes():
    assert check_files(complete(), "xslm-70m") == []


def test_a_missing_quantization_is_reported():
    """Both models publish a Q6_K file, so both must be reported."""
    files = [f for f in complete() if "Q6_K" not in f]
    assert check_files(files, "xslm-70m") == [
        "gguf/xslm-70m-Q6_K.gguf",
        "gguf/xslm-70m-instruct-Q6_K.gguf",
    ]


def test_a_missing_weight_file_is_reported():
    files = [f for f in complete() if f != "model.safetensors"]
    assert "model.safetensors" in check_files(files, "xslm-70m")


def test_an_empty_repository_reports_every_file():
    assert len(check_files([], "xslm-70m")) == len(REQUIRED)


def test_the_required_list_names_every_gguf_file():
    names = " ".join(REQUIRED)
    for quant in ("BF16", "F16", "Q8_0", "Q6_K", "Q4_K_M"):
        assert f"-{quant}.gguf" in names


def test_the_two_sixteen_bit_files_are_separate_entries():
    """BF16 and F16 are different files. A substring test would confuse them."""
    names = [item for item in REQUIRED if item.endswith(".gguf")]
    # Five files for the base model, and five for the tuned model.
    assert len(names) == 10
    assert len(set(names)) == 10


def test_a_repository_that_holds_only_the_base_model_fails():
    """The watchdog reads this check before it destroys the pod.

    While the tuned model is absent, the check must fail, so the watchdog stops the
    pod and the volume keeps the work. A check that passes on the base model alone
    would terminate the pod and lose the tuned weights.
    """
    base_only = [name for name in complete() if "instruct" not in name]
    missing = check_files(base_only, "xslm-70m")
    assert missing
    assert all("instruct" in name for name in missing)


def test_the_tuned_model_carries_its_tokenizer_config():
    """The chat template lives in tokenizer_config.json.

    Without that file llama.cpp guesses the template, which is what made the base
    model answer with punctuation.
    """
    assert "instruct/tokenizer_config.json" in REQUIRED
