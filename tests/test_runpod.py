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


def test_every_request_carries_a_user_agent():
    """Cloudflare answers 403 error 1010 when the User-Agent header is absent."""
    request = build_request("GET", "pods", "a-key")
    assert request.get_header("User-agent")


def test_the_stop_request_posts_to_the_stop_endpoint():
    request = build_request("POST", "pods/abc123/stop", "a-key")
    assert request.full_url == "https://rest.runpod.io/v1/pods/abc123/stop"
    assert request.get_method() == "POST"


def test_the_ssh_string_is_built_from_the_pod_details():
    from runpod import ssh_string_for

    pod = {"publicIp": "1.2.3.4", "portMappings": {"22": 12152}}
    assert ssh_string_for(pod) == "root@1.2.3.4 -p 12152 -i ~/.ssh/id_ed25519"


def test_a_pod_without_an_exposed_port_is_reported():
    from runpod import ssh_string_for

    with pytest.raises(SystemExit, match="TCP port 22"):
        ssh_string_for({"publicIp": "1.2.3.4", "portMappings": {}})
