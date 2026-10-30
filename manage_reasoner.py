"""Pause or resume the reasoner and write a restart receipt."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["pause", "resume"])
parser.add_argument("--state", type=Path, default=root / "reasoner-state.json")
args = parser.parse_args()
state = args.state

if args.action == "pause":
    if state.exists():
        raise SystemExit("A service receipt already exists; inspect it before pausing again.")
    metrics = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10).read().decode()
    for line in metrics.splitlines():
        if line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting")):
            if float(line.rsplit(None, 1)[-1]) > 0:
                raise SystemExit("Reasoner has active/queued requests; not stopping it.")
    matches = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = p.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
            if "/home/ubuntu/env_vllm/bin/vllm" in argv and "nvidia/Cosmos-Reason2-8B" in argv:
                matches.append((p, argv))
        except (OSError, UnicodeError):
            pass
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one known FDE reasoner; found {len(matches)}")
    p, argv = matches[0]
    record = {"pid": int(p.name), "argv": argv, "cwd": str(p.joinpath("cwd").resolve()),
              "paused_at": time.time()}
    state.write_text(json.dumps(record, indent=2) + "\n")
    os.kill(record["pid"], signal.SIGTERM)
    for _ in range(90):
        if not p.exists():
            break
        time.sleep(.5)
    else:
        raise SystemExit("Reasoner did not exit after SIGTERM; no force kill performed.")
    print("Paused idle reasoner; restart receipt saved.", flush=True)
else:
    record = json.loads(state.read_text())
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)
    except Exception:
        env = os.environ.copy()
        service_bin = str(Path(record["argv"][0]).parent)
        env["PATH"] = service_bin + ":" + str(Path.home() / ".local/bin") + ":" + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(Path(service_bin).parent)
        token = Path.home() / ".hf_token"
        if token.exists():
            env["HF_TOKEN"] = token.read_text().strip()
        with (root / "reasoner-restored.log").open("ab") as log:
            process = subprocess.Popen(record["argv"], cwd=record["cwd"], env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(restored_pid=process.pid, restored_at=time.time())
        state.write_text(json.dumps(record, indent=2) + "\n")
        print("Reasoner restart launched", process.pid, flush=True)
    else:
        print("Reasoner already responds to health check.", flush=True)
