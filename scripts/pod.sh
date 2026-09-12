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

if [ -z "${POD_SSH:-}" ]; then
    echo "POD_SSH is empty. Write it in .env, or export it." >&2
    echo 'Example: POD_SSH="root@1.2.3.4 -p 40123 -i ~/.ssh/id_ed25519"' >&2
    exit 1
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
        -e "ssh $SSH_FLAGS" \
        ./ "${POD_SSH%% *}:$REMOTE_DIR/"
    ;;
setup)
    remote "command -v $UV || curl -LsSf https://astral.sh/uv/install.sh | sh"
    remote "cd $REMOTE_DIR && $UV sync"
    remote "cd $REMOTE_DIR && $UV run pytest -q"
    ;;
run)
    shift
    remote "cd $REMOTE_DIR && mkdir -p out && tmux new-session -d -s train \"$UV run $* 2>&1 | tee out/train.log\""
    echo "Started in tmux. Read it with: scripts/pod.sh watch"
    ;;
watch)
    remote "tail -n ${2:-20} $REMOTE_DIR/out/train.log"
    ;;
*)
    echo "Usage: pod.sh {check|sync|setup|run <command>|watch [lines]}" >&2
    exit 1
    ;;
esac
