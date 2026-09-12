import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _env import optional, require  # noqa: E402


def test_require_returns_the_value(monkeypatch):
    monkeypatch.setenv("XSLM_TEST_KEY", "value")
    assert require("XSLM_TEST_KEY") == "value"


def test_require_raises_when_the_value_is_absent(monkeypatch):
    monkeypatch.delenv("XSLM_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="XSLM_TEST_KEY"):
        require("XSLM_TEST_KEY")


def test_the_message_names_the_fix(monkeypatch):
    monkeypatch.delenv("XSLM_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="--env-file"):
        require("XSLM_TEST_KEY")


def test_the_message_never_holds_a_value(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-value-do-not-print")
    monkeypatch.delenv("XSLM_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError) as error:
        require("XSLM_TEST_KEY")
    assert "secret-value-do-not-print" not in str(error.value)


def test_optional_returns_the_default(monkeypatch):
    monkeypatch.delenv("XSLM_TEST_KEY", raising=False)
    assert optional("XSLM_TEST_KEY", "fallback") == "fallback"
