#!/usr/bin/env bash
# Stop the pod from billing after a deadline, even when no agent is watching.
#
# The watchdog runs on the pod itself, so it works when the laptop sleeps and the
# agent session ends.
#
# On the deadline it asks verify_upload.py whether every file reached the Hub:
#   every file present -> terminate the pod, because nothing is left to lose
#   any file absent    -> stop the pod, which ends the GPU charge and keeps the volume
set -euo pipefail

cd "$(dirname "$0")/.."
UV="${UV:-/root/.local/bin/uv}"
DEADLINE_MINUTES="${1:-300}"
STARTED_AT=$(date +%s)
LIMIT=$((DEADLINE_MINUTES * 60))

echo "Watchdog armed. The pod stops after $DEADLINE_MINUTES minutes."

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - STARTED_AT))

    if [ -f out/PIPELINE_DONE ] || [ "$ELAPSED" -ge "$LIMIT" ]; then
        echo "Watchdog firing after $((ELAPSED / 60)) minutes."
        if "$UV" run --env-file .env scripts/verify_upload.py; then
            echo "Every file is on the Hub. Terminating the pod."
            "$UV" run --env-file .env scripts/runpod.py terminate --yes
        else
            echo "The upload is incomplete. Stopping the pod and keeping the volume."
            "$UV" run --env-file .env scripts/runpod.py stop
        fi
        exit 0
    fi

    sleep 60
done
