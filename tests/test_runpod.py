import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from runpod import build_request, guard_terminate  # noqa: E402


def test_the_list_request_targets_the_pods_endpoint():
    request = build_request("GET", "pods", "a-key")
    assert request.full_url == "https://rest.runpod.io/v1/pods"
    assert request.get_method() == "GET"


def test_the_terminate_request_deletes_one_pod():
    request = build_request("DELETE", "pods/abc123", "a-key")
    assert request.full_url == "https://rest.runpod.io/v1/pods/abc123"
    assert request.get_method() == "DELETE"


def test_every_request_carries_the_bearer_token():
    request = build_request("GET", "pods", "a-key")
    assert request.get_header("Authorization") == "Bearer a-key"


def test_terminate_refuses_without_the_confirmation():
    with pytest.raises(SystemExit, match="--yes"):
        guard_terminate("abc123", confirmed=False)


def test_terminate_proceeds_with_the_confirmation():
    assert guard_terminate("abc123", confirmed=True) == "abc123"


def test_terminate_refuses_an_empty_pod_id():
    with pytest.raises(SystemExit, match="RUNPOD_POD_ID"):
        guard_terminate("", confirmed=True)
