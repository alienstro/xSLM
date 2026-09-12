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


def test_quantize_gives_the_converter_the_vocabulary_modules():
    """The converter imports sentencepiece before it reads the vocabulary.

    It catches a missing tokenizer.model and falls back to the BPE vocabulary, but
    a missing module raises ModuleNotFoundError, which no handler catches.
    """
    text = (SCRIPTS / "quantize.sh").read_text()
    assert text.count("--with sentencepiece") == 2
    assert text.count("--with protobuf") == 2


def test_quantize_writes_both_sixteen_bit_files():
    """BF16 keeps the training dtype. F16 serves the tools that read no BF16."""
    text = (SCRIPTS / "quantize.sh").read_text()
    assert "--outtype bf16" in text
    assert "--outtype f16" in text
    assert "$NAME-BF16.gguf" in text
    assert "$NAME-F16.gguf" in text


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


def test_pod_reads_the_env_file():
    text = (SCRIPTS / "pod.sh").read_text()
    assert "set -a" in text
    assert ".env" in text


def test_the_env_example_names_the_pod_connection():
    example = (SCRIPTS.parent / ".env.example").read_text()
    assert "POD_SSH" in example


def test_the_env_example_names_every_variable_the_scripts_read():
    example = (SCRIPTS.parent / ".env.example").read_text()
    for name in ("HF_TOKEN", "HF_REPO_ID", "MODEL_LICENSE", "WANDB_API_KEY",
                 "WANDB_PROJECT", "POD_SSH"):
        assert f"{name}=" in example


def test_pod_strips_a_pasted_ssh_prefix():
    """The RunPod console shows a full command, so POD_SSH often starts with 'ssh '."""
    text = (SCRIPTS / "pod.sh").read_text()
    assert "POD_SSH#ssh " in text


def test_pod_gives_rsync_the_port_and_the_key():
    """POD_SSH holds -p and -i after the host. rsync needs them inside its -e string."""
    text = (SCRIPTS / "pod.sh").read_text()
    assert "SSH_EXTRA" in text
    assert '-e "ssh $SSH_FLAGS $SSH_EXTRA"' in text


def test_pod_can_copy_the_env_file_to_the_pod():
    text = (SCRIPTS / "pod.sh").read_text()
    assert "secrets)" in text
    assert "chmod 600" in text


def test_pod_names_each_tmux_session_after_its_command():
    """A fixed session name would collide with a job that is still running."""
    text = (SCRIPTS / "pod.sh").read_text()
    assert "-s \\\"$SESSION\\\"" in text


def test_pod_passes_the_env_file_to_uv_when_it_exists():
    assert "--env-file .env" in (SCRIPTS / "pod.sh").read_text()


def test_pod_watch_turns_carriage_returns_into_lines():
    """tqdm writes with \\r, so a plain tail returns one enormous line."""
    assert "tr '\\\\r' '\\\\n'" in (SCRIPTS / "pod.sh").read_text()


def test_the_env_example_names_the_runpod_key():
    example = (SCRIPTS.parent / ".env.example").read_text()
    assert "RUNPOD_API_KEY=" in example
    assert "RUNPOD_POD_ID=" in example


def test_pod_offers_pods_and_terminate():
    text = (SCRIPTS / "pod.sh").read_text()
    assert "pods)" in text
    assert "terminate)" in text


def test_pod_terminate_demands_the_confirmation():
    """A terminate that runs on its own would destroy an unpublished model."""
    text = (SCRIPTS / "pod.sh").read_text()
    assert "--yes" in text


def test_pod_offers_the_upload_verification():
    assert "verify)" in (SCRIPTS / "pod.sh").read_text()


def test_the_watchdog_stops_the_pod_when_the_upload_is_incomplete():
    """A terminate before the push would destroy the only copy of the model."""
    text = (SCRIPTS / "watchdog.sh").read_text()
    assert "verify_upload.py" in text
    assert "runpod.py stop" in text
    assert "runpod.py terminate --yes" in text
    subprocess.run(["bash", "-n", str(SCRIPTS / "watchdog.sh")], check=True)


def test_the_data_pipeline_skips_a_tokenizer_that_exists():
    text = (SCRIPTS / "run_data_pipeline.sh").read_text()
    assert "data/tokenizer.json" in text
    subprocess.run(["bash", "-n", str(SCRIPTS / "run_data_pipeline.sh")], check=True)


def test_pod_names_the_session_after_the_script_not_the_interpreter():
    """'run bash scripts/x.sh' must name the session x, not bash."""
    assert "*.sh|*.py)" in (SCRIPTS / "pod.sh").read_text()


def test_the_watchdog_stops_the_pod_when_the_agent_stops_checking_in():
    """A dead agent means nobody will push the model, so the pod must stop."""
    text = (SCRIPTS / "watchdog.sh").read_text()
    assert "HEARTBEAT" in text
    assert "HEARTBEAT_GRACE" in text


def test_pod_refreshes_the_heartbeat_when_it_reads_the_pod():
    """The heartbeat must refresh on its own, not depend on the agent remembering."""
    text = (SCRIPTS / "pod.sh").read_text()
    assert "heartbeat)" in text
    assert "touch $REMOTE_DIR/out/HEARTBEAT" in text


def test_pod_can_repair_a_stale_connection_string():
    """RunPod maps a new public port on every container restart."""
    text = (SCRIPTS / "pod.sh").read_text()
    assert "reconnect)" in text
    assert "runpod.py ssh" in text
