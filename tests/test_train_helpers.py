import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train import compute_max_steps, resolve_report_to  # noqa: E402


def test_the_step_count_follows_the_throughput():
    assert compute_max_steps(63_000.0, 10_800.0, 262_144) == 2595


def test_the_step_count_is_never_zero():
    assert compute_max_steps(1.0, 1.0, 262_144) == 1


def test_weights_and_biases_is_on_when_the_key_exists(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "a-value")
    assert resolve_report_to() == ["wandb"]


def test_weights_and_biases_is_off_when_the_key_is_absent(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert resolve_report_to() == []
