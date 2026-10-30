"""Run the official CDF Augmentation image sequentially; record real exit codes."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--only", help="one job id, for smoke tests")
parser.add_argument("--stage", choices=["clean", "bootstrap"])
args = parser.parse_args()
image = json.loads((ROOT / "results/runtime-images.json").read_text())["augmentation"]["id"]
container = "cdf-route1-augmentation"
existing = subprocess.run(["docker", "inspect", container], capture_output=True, text=True)
if existing.returncode:
    subprocess.run(["docker", "run", "-d", "--name", container, "--network", "cdf-route1",
        "--user", "1000:1000", "-v", f"{ROOT}:{ROOT}", "--entrypoint", "/bin/bash", image,
        "-c", "sleep infinity"], check=True)
else:
    info = json.loads(existing.stdout)[0]
    if info["Config"]["Image"] != image:
        raise SystemExit("Existing CDF edit container has a different pinned image")
    if not info["State"]["Running"]:
        subprocess.run(["docker", "start", container], check=True)
for job in json.loads((ROOT / "jobs.json").read_text()):
    if args.only and job["id"] != args.only:
        continue
    if args.stage and job["stage"] != args.stage:
        continue
    receipt_path = ROOT / "results" / (job["id"] + "-receipt.json")
    if receipt_path.exists() and json.loads(receipt_path.read_text()).get("exit_code") == 0 and Path(job["output"]).exists():
        print("Already complete:", job["id"], flush=True)
        continue
    command = ["docker", "exec", container, "uv", "run", "--no-sync", "/workspace/modules/cli.py",
        "--config", job["config"]]
    effective_config = json.loads(Path(job["config"]).read_text())
    started = time.time()
    print("Starting", job["id"], flush=True)
    with (ROOT / "logs" / (job["id"] + ".log")).open("w") as log:
        valid = subprocess.run(command + ["--validate-only"], stdout=log, stderr=subprocess.STDOUT)
        if valid.returncode:
            raise SystemExit(f"CDF config validation failed: {job['id']}")
        run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    receipt = {"job": job, "effective_config": effective_config,
        "command": command, "image_id": image, "exit_code": run.returncode,
        "elapsed_seconds": time.time() - started, "output_exists": Path(job["output"]).exists()}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    if run.returncode or not receipt["output_exists"]:
        raise SystemExit(f"CDF image-edit failed: {job['id']}")
    print("Completed", job["id"], round(receipt["elapsed_seconds"], 1), "seconds", flush=True)
