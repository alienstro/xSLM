"""Check that the HuggingFace repository holds every published file.

The pod disk holds the only copy of the model until this check passes, because the
pod has no volume disk. Terminate the pod only after this script exits with zero.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import require  # noqa: E402

from xslm.config import load_yaml  # noqa: E402

REQUIRED = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "README.md",
    "gguf/{name}-BF16.gguf",
    "gguf/{name}-F16.gguf",
    "gguf/{name}-Q8_0.gguf",
    "gguf/{name}-Q6_K.gguf",
    "gguf/{name}-Q4_K_M.gguf",
    # The tuned model. The watchdog reads this list before it destroys the pod, so
    # a missing tuned file must fail the check and leave the pod stopped instead.
    "instruct/config.json",
    "instruct/model.safetensors",
    "instruct/tokenizer.json",
    # The chat template lives here, and llama.cpp guesses without it.
    "instruct/tokenizer_config.json",
    "gguf/{name}-instruct-BF16.gguf",
    "gguf/{name}-instruct-F16.gguf",
    "gguf/{name}-instruct-Q8_0.gguf",
    "gguf/{name}-instruct-Q6_K.gguf",
    "gguf/{name}-instruct-Q4_K_M.gguf",
]


def check_files(present, model_name):
    """Return the required files that the repository does not hold."""
    have = set(present)
    return [item.format(name=model_name) for item in REQUIRED if item.format(name=model_name) not in have]


def main():
    token = require("HF_TOKEN")
    repo_id = require("HF_REPO_ID")
    model_name = load_yaml()["publish"]["model_name"]

    files = HfApi(token=token).list_repo_files(repo_id, repo_type="model")
    missing = check_files(files, model_name)

    for name in sorted(files):
        print(f"  present: {name}")
    if missing:
        print(f"\nMISSING {len(missing)} file(s):")
        for name in missing:
            print(f"  missing: {name}")
        raise SystemExit(1)
    print(f"\nEvery required file is on {repo_id}. The pod is safe to terminate.")


if __name__ == "__main__":
    main()
