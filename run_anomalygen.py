"""Run AnomalyGen training, generation, evaluation, and labeling."""
import csv
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--prepare-only", action="store_true")
parser.add_argument("--start-at", choices=["train", "wait", "generate", "evaluate", "refine", "label"], default="train",
    help="Resume a stage without retraining; wait attaches to an already running official training job")
parser.add_argument("--run-name", default="excavator_route1")
parser.add_argument("--recipe", type=Path, default=root / "configs/train.yaml")
parser.add_argument("--dataset-dir", type=Path, default=root / "dataset")
parser.add_argument("--generate-jsonl", type=Path, default=root / "results/target-masks/testcase.jsonl")
parser.add_argument("--output-root", type=Path, default=root / "results/anomalygen")
args = parser.parse_args()
image = json.loads((root / "results/runtime-images.json").read_text())["anomalygen"]["id"]
provision = json.loads((root / "results/anomaly-provision.json").read_text())
if provision["exit_code"] != 0:
    raise SystemExit("Official AnomalyGen checkpoint provisioning has not passed")
name = "cdf-route1-anomaly"
run_name = args.run_name
recipe_path = args.recipe.resolve()
dataset_dir = args.dataset_dir.resolve()
generate_jsonl = args.generate_jsonl.resolve()
output_root = args.output_root.resolve()
env = os.environ.copy()
env["HF_TOKEN"] = (Path.home() / ".hf_token").read_text().strip()
start = ["docker", "run", "-d", "--name", name, "--gpus", "all", "--shm-size", "4g",
    "--network", "cdf-route1", "--user", "1000:1000", "-e", "HF_TOKEN", "-e", "UV_CACHE_DIR=/tmp/uv-cache",
    "-v", f"{root}:{root}",
    "-v", f"{root}/checkpoints:/workspace/cdf-anomalygen/checkpoints",
    "-v", f"{Path.home()}/.cache/huggingface/hub:/workspace/cdf-anomalygen/checkpoints/hf",
    "-v", f"{root}/anomaly-results:/workspace/cdf-anomalygen/results",
    "--entrypoint", "/bin/bash", image, "-c", "sleep infinity"]
receipts = root / "results/anomaly-stages.jsonl"

def run(stage, args, allowed=(0,)):
    command = ["docker", "exec", name, *args]
    started = time.time()
    print(stage, flush=True)
    with (root / "logs" / (stage + ".log")).open("a") as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
    with receipts.open("a") as f:
        f.write(json.dumps({"stage": stage, "command": command, "image_id": image,
            "exit_code": result.returncode, "elapsed_seconds": time.time() - started}) + "\n")
    if result.returncode not in allowed:
        raise RuntimeError(f"{stage} failed with exit {result.returncode}; inspect logs/{stage}.log")
    return result.returncode

existing = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
if existing.returncode:
    subprocess.run(start, env=env, check=True)
else:
    info = json.loads(existing.stdout)[0]
    if info["Config"]["Image"] != image:
        raise SystemExit("Existing AnomalyGen container has a different pinned image")
    if not info["State"]["Running"]:
        subprocess.run(["docker", "start", name], check=True)
if args.prepare_only:
    run("anomaly-runtime-smoke", ["python", "-c",
        "import torch, anomalygen; from anomalygen.configs.texture import exp_config; "
        "x=torch.ones((64,64),device='cuda',dtype=torch.bfloat16); "
        "assert float((x@x)[0,0]) == 64; "
        "print('AnomalyGen imports and CUDA BF16 matmul passed:',torch.cuda.get_device_name(0),torch.__version__)"])
    print("AnomalyGen runtime ready; training has not started.", flush=True)
    raise SystemExit(0)
try:
    if args.start_at == "train":
        run("anomaly-train-start", ["bash", "scripts/skill_utility/train_control.sh", "start",
            "--name", run_name, "--recipe", str(recipe_path), "--num_gpus", "1",
            "--scripts", "anomalygen/scripts/texture", "--task", "texture_ft"])
    if args.start_at in {"train", "wait"}:
        while True:
            code = run("anomaly-train-wait", ["bash", "scripts/skill_utility/train_control.sh", "wait",
                "--name", run_name, "--timeout", "45"], allowed=(0, 2))
            if code == 0:
                break
    host_run = root / "anomaly-results/anomalygen" / run_name / "anomalygen_texture_ft"
    container_run = Path("/workspace/cdf-anomalygen/results/anomalygen") / run_name / "anomalygen_texture_ft"
    best = (host_run / "checkpoints/best_checkpoint.txt").read_text().strip()
    checkpoint = str(container_run / "checkpoints/model" / best)
    # config.yaml is the framework's expanded config, not an AnomalyGen recipe.
    saved_recipe = host_run / "train.yaml"
    if not saved_recipe.exists():
        copies = sorted(host_run.glob("train_*.yaml"))
        if len(copies) != 1:
            raise RuntimeError("TrainingReport did not save its train.yaml recipe")
        saved_recipe = copies[0]
    recipe = str(container_run / saved_recipe.name)
    stage_order = ["train", "wait", "generate", "evaluate", "refine", "label"]
    def needed(stage):
        return stage_order.index(args.start_at) <= stage_order.index(stage)
    sdg = output_root / "sdg"
    sdg.mkdir(parents=True, exist_ok=True)
    scripts = "anomalygen/scripts/texture/"
    if needed("generate"):
        run("anomaly-generate", ["torchrun", "--nproc_per_node=1", scripts + "generate.py",
        "--checkpoint", checkpoint, "--recipe", recipe,
        "--input_data_path", str(generate_jsonl), "--output_dir", str(sdg),
        "--offload_guardrail_models"])
    if needed("evaluate"):
        run("anomaly-evaluate", ["python", scripts + "evaluate.py", "--gen_root", str(sdg),
        "--real_root", str(dataset_dir), "--recipe", recipe, "--output_file", str(sdg / "kpi.json")])
    searched = output_root / "searched"
    if needed("refine"):
        run("anomaly-refine", ["python", scripts + "quality_refine.py", "run",
        "--base_testcase", str(generate_jsonl),
        "--original", str(sdg), "--original_kpi", str(sdg / "kpi.json"),
        "--rounds_dir", str(output_root / "rounds"), "--output", str(searched),
        "--final_kpi", str(searched / "kpi.json"), "--checkpoint", checkpoint,
        "--recipe", recipe, "--real_root", str(dataset_dir),
        "--score", "nn", "--num_search_run", "1", "--num_gpus", "1"])
    if needed("label"):
        run("anomaly-pseudo-label", ["python", scripts + "pseudo_label.py", "--gen_root", str(searched),
        "--output_dir", str(searched / "pseudo_labels"), "--no_caption"])
    mapping = {}
    with (searched / "texture_ft_generation_result.csv").open() as f:
        for row in csv.DictReader(f):
            mapping[Path(row["image_filename"]).stem] = str(
                (searched / "reconstructed_image" / row["output_filename"]).relative_to(root))
    (output_root / "output-map.json").write_text(json.dumps(mapping, indent=2))
    print("AnomalyGen stages completed; visual review is still required.", flush=True)
finally:
    subprocess.run(["docker", "stop", name], check=False)
