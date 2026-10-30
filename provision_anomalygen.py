"""Download the Nano weights used by this pilot."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
tag = "nvcr.io/nvidia/cdf-anomalygen:1.2.0"
for attempt in range(180):
    check = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, text=True)
    if check.returncode == 0:
        info = json.loads(check.stdout)[0]
        break
    time.sleep(10)
else:
    raise SystemExit("AnomalyGen runtime not available after 30 minutes")
record_path = root / "results/runtime-images.json"
record = json.loads(record_path.read_text()) if record_path.exists() else {}
record["anomalygen"] = {"id": info["Id"], "repo_digests": info["RepoDigests"],
    "size": info["Size"], "labels": info["Config"].get("Labels")}
record_path.write_text(json.dumps(record, indent=2))
for folder in ("checkpoints", "anomaly-results"):
    (root / folder).mkdir(exist_ok=True)
env = os.environ.copy()
env["HF_TOKEN"] = (Path.home() / ".hf_token").read_text().strip()
cmd = ["docker", "run", "--rm", "--name", "cdf-route1-provision", "--gpus", "all",
    "--user", "1000:1000", "-e", "HF_TOKEN", "-e", "UV_CACHE_DIR=/tmp/uv-cache",
    "-v", f"{root}/checkpoints:/workspace/cdf-anomalygen/checkpoints",
    "-v", f"{Path.home()}/.cache/huggingface/hub:/workspace/cdf-anomalygen/checkpoints/hf",
    "--entrypoint", "/bin/bash", info["Id"], "-c",
    "bash scripts/download_checkpoints.sh base-nano dinov2 cradio wan-vae sam2 guardrail qwen-tokenizer"]
print("Provisioning selected official AnomalyGen checkpoint groups", flush=True)
result = subprocess.run(cmd, env=env)
(root / "results/anomaly-provision.json").write_text(json.dumps({
    "command": cmd, "exit_code": result.returncode, "runtime_id": info["Id"],
}, indent=2))
raise SystemExit(result.returncode)
