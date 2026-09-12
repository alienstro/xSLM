#!/usr/bin/env bash
# The SSH and rsync helper for the RunPod pod.
#
# Set POD_SSH to the connection string that the RunPod console shows, for example:
#     export POD_SSH="root@213.173.98.10 -p 40123 -i ~/.ssh/id_ed25519"
set -euo pipefail

REMOTE_DIR="/workspace/xSLM"
SSH_FLAGS="-o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new"
UV="/root/.local/bin/uv"

# Read .env, so that one file holds every setting. An exported value wins over
# the file, which lets the caller point at a second pod for one command.
ENV_FILE="${ENV_FILE:-.env}"
if [ -f "$ENV_FILE" ] && [ -z "${POD_SSH:-}" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi

# The RunPod console shows a whole command, so the pasted value often starts with
# "ssh ". Strip that word, or every later command would treat it as the hostname.
POD_SSH="${POD_SSH#ssh }"

if [ -z "${POD_SSH:-}" ]; then
    echo "POD_SSH is empty. Write it in .env, or export it." >&2
    echo 'Example: POD_SSH="root@1.2.3.4 -p 40123 -i ~/.ssh/id_ed25519"' >&2
    exit 1
fi

# POD_SSH holds the host first, then flags such as "-p 40123 -i ~/.ssh/id_ed25519".
# ssh takes the whole string, but rsync needs the host alone and the flags inside -e.
SSH_HOST="${POD_SSH%% *}"
SSH_EXTRA=""
SCP_FLAGS=""
if [ "$SSH_HOST" != "$POD_SSH" ]; then
    SSH_EXTRA="${POD_SSH#* }"
    # scp spells the port -P, not -p.
    SCP_FLAGS="$(echo "$SSH_EXTRA" | sed "s/-p /-P /")"
fi

remote() {
    # shellcheck disable=SC2086
    ssh $SSH_FLAGS $POD_SSH "$@"
}

case "${1:-}" in
check)
    remote 'nvidia-smi'
    remote 'cmake --version || (apt-get update -qq && apt-get install -y -qq cmake)'
    remote 'tmux -V || (apt-get update -qq && apt-get install -y -qq tmux)'
    remote 'rsync --version | head -1 || (apt-get update -qq && apt-get install -y -qq rsync)'
    ;;
sync)
    remote "mkdir -p $REMOTE_DIR"
    # shellcheck disable=SC2086
    rsync -az --delete \
        --exclude '.git' --exclude '.venv' --exclude 'data' --exclude 'out' \
        --exclude 'llama.cpp' --exclude '__pycache__' --exclude '.env' \
        -e "ssh $SSH_FLAGS $SSH_EXTRA" \
        ./ "$SSH_HOST:$REMOTE_DIR/"
    ;;
setup)
    # Keep the caches on the volume, so a container restart does not download
    # the CUDA wheels again.
    remote "command -v $UV || curl -LsSf https://astral.sh/uv/install.sh | sh"
    remote "grep -q UV_CACHE_DIR /root/.bashrc || printf 'export UV_CACHE_DIR=/workspace/.uv-cache\nexport HF_HOME=/workspace/.hf\n' >> /root/.bashrc"
    remote "cd $REMOTE_DIR && UV_CACHE_DIR=/workspace/.uv-cache $UV sync"
    remote "cd $REMOTE_DIR && $UV run pytest -q"
    ;;
secrets)
    # The RunPod template held no environment fields, so copy .env to the pod.
    # The file dies when the pod is terminated.
    scp $SSH_FLAGS $SCP_FLAGS "$ENV_FILE" "$SSH_HOST:$REMOTE_DIR/.env"
    remote "chmod 600 $REMOTE_DIR/.env && echo 'Wrote .env with mode 600.'"
    ;;
run)
    shift
    # Name the session after the command, so a second job cannot collide with a
    # job that is still running.
    SESSION="$(basename "${1%.*}")"
    LOG="out/$SESSION.log"
    remote "cd $REMOTE_DIR && mkdir -p out && tmux new-session -d -s \"$SESSION\" \"$UV run --env-file .env $* 2>&1 | tee $LOG\""
    echo "Started tmux session '$SESSION'. Read it with: scripts/pod.sh watch $SESSION"
    ;;
watch)
    # tqdm writes progress with a carriage return, so one line can hold megabytes.
    # Turn every carriage return into a newline before tail reads the file.
    remote "tr '\\r' '\\n' < $REMOTE_DIR/out/${2:-train}.log | grep -v '^\$' | tail -n ${3:-20}"
    ;;
sessions)
    remote "tmux list-sessions 2>/dev/null || echo 'no tmux session'"
    ;;
verify)
    # The pod disk holds the only copy of the model until this passes.
    uv run --env-file "$ENV_FILE" scripts/verify_upload.py
    ;;
pods)
    uv run --env-file "$ENV_FILE" scripts/runpod.py list
    ;;
terminate)
    # Warning: this destroys the pod disk and cannot be undone. Never run it
    # before the HuggingFace push is verified.
    shift
    uv run --env-file "$ENV_FILE" scripts/runpod.py terminate "$@"
    ;;
*)
    echo "Usage: pod.sh {check|sync|secrets|setup|run <command>|watch [name] [lines]|sessions|verify|pods|terminate --yes}" >&2
    exit 1
    ;;
esac
