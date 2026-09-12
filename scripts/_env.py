"""Read the environment, and fail in one second instead of after 30 minutes."""

import os

FIX = (
    "Copy .env.example to .env, write the value, and run the script as\n"
    "    uv run --env-file .env scripts/<name>.py\n"
    "On the pod, set the value in the RunPod template environment fields instead."
)


def require(name):
    """Return the value of one environment variable, or raise."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"The environment variable {name} is empty or absent.\n{FIX}")
    return value


def optional(name, default=None):
    """Return the value of one environment variable, or the default."""
    return os.environ.get(name) or default
