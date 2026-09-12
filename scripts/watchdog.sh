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
HEARTBEAT_GRACE="${2:-30}"
STARTED_AT=$(date +%s)
LIMIT=$((DEADLINE_MINUTES * 60))
GRACE=$((HEARTBEAT_GRACE * 60))
HEARTBEAT=out/HEARTBEAT

mkdir -p out
touch "$HEARTBEAT"

echo "Watchdog armed. The pod stops after $DEADLINE_MINUTES minutes,"
echo "or $HEARTBEAT_GRACE minutes after the agent stops checking in."

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - STARTED_AT))
    SILENT=$((NOW - $(stat -c %Y "$HEARTBEAT")))
    REASON=""

    [ -f out/PIPELINE_DONE ] && REASON="the pipeline finished"
    [ "$ELAPSED" -ge "$LIMIT" ] && REASON="the $DEADLINE_MINUTES minute deadline passed"
    [ "$SILENT" -ge "$GRACE" ] && REASON="the agent has been silent for $((SILENT / 60)) minutes"

    if [ -n "$REASON" ]; then
        echo "Watchdog firing after $((ELAPSED / 60)) minutes, because $REASON."
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
