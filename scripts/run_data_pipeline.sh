#!/usr/bin/env bash
# Steps 1 and 2 in one session: train the tokenizer, then write the token files.
#
# Both steps run in one tmux session, so the whole preparation survives a closed
# laptop and an ended agent session.
set -euo pipefail

cd "$(dirname "$0")/.."
UV="${UV:-/root/.local/bin/uv}"

if [ ! -f data/tokenizer.json ]; then
    "$UV" run --env-file .env scripts/train_tokenizer.py
else
    echo "data/tokenizer.json exists, so step 1 is skipped."
fi

"$UV" run --env-file .env scripts/prepare_data.py

touch out/DATA_READY
echo "Steps 1 and 2 are complete."
