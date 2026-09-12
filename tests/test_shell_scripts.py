import subprocess
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_quantize_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPTS / "quantize.sh")], check=True)


def test_quantize_stops_on_an_error():
    text = (SCRIPTS / "quantize.sh").read_text()
    assert "set -euo pipefail" in text


def test_quantize_builds_every_requested_type():
    text = (SCRIPTS / "quantize.sh").read_text()
    for quant in ("Q8_0", "Q6_K", "Q4_K_M"):
        assert quant in text


def test_pod_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPTS / "pod.sh")], check=True)


def test_pod_keeps_the_connection_alive():
    assert "ServerAliveInterval=30" in (SCRIPTS / "pod.sh").read_text()


def test_pod_starts_the_long_run_inside_tmux():
    text = (SCRIPTS / "pod.sh").read_text()
    assert "tmux new-session -d" in text


def test_pod_checks_for_cmake_before_the_run():
    assert "cmake --version" in (SCRIPTS / "pod.sh").read_text()


def test_pod_never_sends_the_env_file_to_the_pod():
    assert "--exclude '.env'" in (SCRIPTS / "pod.sh").read_text()
