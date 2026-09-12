"""Read and terminate RunPod pods through the REST API.

The design spec first configured no RunPod API key, so that no script could stop a
pod. The person later asked for a terminate command, to be sure that no pod keeps
billing after the run. Terminate is therefore explicit: it needs --yes, and it never
runs as a step of the pipeline.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

BASE_URL = "https://rest.runpod.io/v1"


def build_request(method, path, token, payload=None):
    """Return one urllib request for the RunPod REST API."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(f"{BASE_URL}/{path}", data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Content-Type", "application/json")
    # Cloudflare answers 403 error 1010 when the User-Agent header is absent.
    request.add_header("User-Agent", "xslm/0.1 (+https://github.com/)")
    return request


def send(request):
    """Send one request and return the decoded body, or an empty dictionary."""
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode()
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode()[:400]
        raise SystemExit(f"RunPod answered {error.code}: {detail}") from error


def guard_terminate(pod_id, confirmed):
    """Stop unless the caller named a pod and passed the confirmation."""
    if not pod_id:
        raise SystemExit("RUNPOD_POD_ID is empty. Run 'list' to find the identifier.")
    if not confirmed:
        raise SystemExit(
            f"Terminating {pod_id} destroys its disk and cannot be undone.\n"
            "Pass --yes to confirm."
        )
    return pod_id


def list_pods(token):
    """Print one line for each pod: identifier, name, state, and cost."""
    payload = send(build_request("GET", "pods", token))
    pods = payload if isinstance(payload, list) else payload.get("data", [])
    if not pods:
        print("No pod is running.")
        return pods
    for pod in pods:
        print(
            f"{pod.get('id')}  {pod.get('name', '?')}  "
            f"{pod.get('desiredStatus', pod.get('status', '?'))}  "
            f"${pod.get('costPerHr', '?')}/hr"
        )
    return pods


def main():
    parser = argparse.ArgumentParser(description="Read and terminate RunPod pods.")
    parser.add_argument("command", choices=["list", "terminate"])
    parser.add_argument("--pod-id", default=None)
    parser.add_argument("--yes", action="store_true", help="Confirm the termination.")
    arguments = parser.parse_args()

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from _env import optional, require

    token = require("RUNPOD_API_KEY")

    if arguments.command == "list":
        list_pods(token)
        return

    pod_id = guard_terminate(arguments.pod_id or optional("RUNPOD_POD_ID", ""), arguments.yes)
    send(build_request("DELETE", f"pods/{pod_id}", token))
    print(f"Terminated {pod_id}. Billing has stopped.")


if __name__ == "__main__":
    main()
