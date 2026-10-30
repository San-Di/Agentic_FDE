"""Run one checkpoint job on the GPU machine."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from studio_catalog import DAMAGES, expand

ROOT = Path(__file__).resolve().parent
CHECKPOINT = (
    "anomaly-results/anomalygen/rental_equipment_damage/"
    "anomalygen_texture_ft/checkpoints/model/iter_000002000.pt"
)
RECIPE = (
    "anomaly-results/anomalygen/rental_equipment_damage/"
    "anomalygen_texture_ft/train_rental_equipment_damage.yaml"
)
CONTAINER_CHECKPOINT = "/workspace/cdf-anomalygen/results/" + CHECKPOINT.split("anomaly-results/", 1)[1]
CONTAINER_RECIPE = "/workspace/cdf-anomalygen/results/" + RECIPE.split("anomaly-results/", 1)[1]
ANOMALY = "cdf-route1-anomaly"
AUGMENT = "cdf-route1-augmentation"
EDIT_URL = "http://172.18.0.1:8011/health"


def emit(out: Path, **event):
    event["t"] = time.time()
    line = json.dumps(event, ensure_ascii=False)
    with (out / "events.jsonl").open("a") as handle:
        handle.write(line + "\n")
    print("EVENT " + line, flush=True)


def say(out: Path, text: str):
    emit(out, kind="message", role="pipeline", text=text)


def run(command, log_path: Path, timeout=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    return completed.returncode


def gpu_mib():
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if completed.returncode:
        return None
    return int(completed.stdout.strip().splitlines()[0])


def wait_gpu(limit=1500, attempts=40):
    for _ in range(attempts):
        used = gpu_mib()
        if used is not None and used < limit:
            return used
        time.sleep(3)
    raise RuntimeError(f"GPU memory stayed above {limit} MiB")


def edit_ready():
    try:
        with urllib.request.urlopen(EDIT_URL, timeout=5) as response:
            return json.load(response).get("status") == "ready"
    except Exception:
        return False


def edit_server_running():
    completed = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True)
    return any("edit_server.py" in line and "python" in line for line in completed.stdout.splitlines())


def wait_until_edit_ready(out: Path, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if edit_ready():
            say(out, "Image edit is ready.")
            return True
        if not edit_server_running():
            return False
        time.sleep(10)
    raise RuntimeError("Image edit did not become ready within 15 minutes.")


def ensure_edit_server(out: Path):
    if edit_ready():
        say(out, "Qwen Image Edit 2511 is already serving. Clean scenes will use it.")
        return None
    if edit_server_running():
        say(out, "Qwen Image Edit 2511 is already loading. Waiting for that copy instead of starting a second one.")
        if wait_until_edit_ready(out):
            return None
    say(out, "Starting Qwen Image Edit 2511. The weights take several minutes to load.")
    log = out / "logs" / "edit-server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w")
    proc = subprocess.Popen(
        ["bash", "start_edit_server.sh"], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("The image-edit server exited before it was ready. See logs/edit-server.log.")
        if edit_ready():
            say(out, "Image edit is ready.")
            return proc
        time.sleep(10)
    os.killpg(proc.pid, signal.SIGTERM)
    raise RuntimeError("Image edit did not become ready within 15 minutes.")


def stop_edit_server(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    else:
        subprocess.run(["pkill", "-f", "edit_server.py --host"], check=False)
    wait_gpu()


def ensure_container(name, image_key):
    image = json.loads((ROOT / "results/runtime-images.json").read_text())[image_key]["id"]
    existing = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
    if existing.returncode:
        subprocess.run([
            "docker", "run", "-d", "--name", name, "--network", "cdf-route1",
            "--user", "1000:1000", "-v", f"{ROOT}:{ROOT}", "--entrypoint", "/bin/bash", image,
            "-c", "sleep infinity",
        ], check=True)
        return
    info = json.loads(existing.stdout)[0]
    if not info["State"]["Running"]:
        subprocess.run(["docker", "start", name], check=True)


def ensure_anomaly():
    existing = subprocess.run(["docker", "inspect", ANOMALY], capture_output=True, text=True)
    if existing.returncode:
        raise RuntimeError("The AnomalyGen container is not on this machine.")
    info = json.loads(existing.stdout)[0]
    if not info["State"]["Running"]:
        subprocess.run(["docker", "start", ANOMALY], check=True)


def write_edit_config(path: Path, source: Path, destination: Path, prompt: str, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "data": [{
            "inputs": {"rgb": str(source)},
            "output": {
                "video": str(destination),
                "caption": str(destination.with_suffix(".txt")),
                "metadata": str(destination.with_suffix(".json")),
            },
        }],
        "endpoints": [{
            "id": "image-edit",
            "role": "image_edit",
            "url": "http://172.18.0.1:8011/v1",
            "model": "Qwen/Qwen-Image-Edit-2511",
            "adapter": "openai.images.edits",
            "timeout": 1200,
        }],
        "captioning": {"llm": {"text": prompt}},
        "augmentation": {"model": {"name": "image-edit"}, "parameters": {"seed": seed, "num_inference_steps": 40}},
        "pipeline": {"retry": 0},
    }, indent=2))


def edit_image(out: Path, config: Path, destination: Path):
    if destination.exists() and destination.stat().st_size > 10000:
        return
    code = run([
        "docker", "exec", AUGMENT, "uv", "run", "--no-sync", "/workspace/modules/cli.py",
        "--config", str(config),
    ], out / "logs" / "edits.log", timeout=1500)
    if code or not destination.exists():
        raise RuntimeError(f"Image edit failed for {destination.name}. See logs/edits.log.")


def link_training_masks(dataset: Path, texture: str, defect: str):
    source = ROOT / "dataset_ft" / texture
    target = dataset / texture
    target.mkdir(parents=True, exist_ok=True)
    for name in ("mask", "anomaly_image"):
        src = source / name
        dst = target / name
        if not src.exists():
            raise RuntimeError(f"Missing training folder {src}")
        if not dst.exists():
            dst.symlink_to(src)
    (target / "clean_image").mkdir(parents=True, exist_ok=True)


def place_masks(out: Path, dataset: Path, defect: dict, count: int):
    spec = out / "amp" / f"{defect['defect']}.jsonl"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(json.dumps({
        "defect_type": defect["anomaly_type"],
        "spatial_dependency": "text",
        "roi_prompt_defect_location": defect["roi_prompt"],
    }) + "\n")
    amp_dir = out / "amp" / defect["defect"]
    amp_dir.mkdir(parents=True, exist_ok=True)
    code = run([
        "docker", "exec", ANOMALY, "python", "-m",
        "anomalygen.scripts.auto_mask_placement.roi_place_pipeline",
        "--num_sdg", str(count),
        "--mode", "inference",
        "--defect_desc", str(spec),
        "--dataset_dir", str(dataset),
        "--output_dir", str(amp_dir),
        "--seed", str(40 + count),
        "--per_defect_counts", json.dumps({defect["anomaly_type"]: count}),
        "--per_clean_pairing", "on",
    ], out / "logs" / f"amp-{defect['defect']}.log", timeout=3600)
    jsonl = amp_dir / "testcase.jsonl"
    if code or not jsonl.exists():
        raise RuntimeError(f"Mask placement failed for {defect['label']}. See logs/amp-{defect['defect']}.log.")
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(
            f"Mask placement found no {defect['label'].lower()} region. "
            "The part locator returned no box on these clean scenes."
        )
    if len(rows) != count:
        say(out, f"{defect['label']}: placed {len(rows)} of {count} masks. The rest had no visible part to mark.")
    return rows


def generate(out: Path, rows):
    jsonl = out / "testcase.jsonl"
    normalized = []
    for index, row in enumerate(rows):
        normalized.append({
            "image_filename": row["image_filename"],
            "mask_filename": row["mask_filename"],
            "anomaly_type": row["anomaly_type"],
            "num_generated_images": 1,
            "guidance": 3.0,
            "num_steps": 35,
            "crop_ratio": 2.0,
            "seed": 9000 + index,
            "index": index,
            "poisson_blend": False,
        })
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in normalized))
    sdg = out / "sdg"
    sdg.mkdir(parents=True, exist_ok=True)
    code = run([
        "docker", "exec", ANOMALY, "torchrun", "--nproc_per_node=1",
        "anomalygen/scripts/texture/generate.py",
        "--checkpoint", CONTAINER_CHECKPOINT,
        "--recipe", CONTAINER_RECIPE,
        "--input_data_path", str(jsonl),
        "--output_dir", str(sdg),
        "--offload_guardrail_models",
    ], out / "logs" / "generate.log", timeout=7200)
    images = sorted((sdg / "reconstructed_image").glob("*.png")) if (sdg / "reconstructed_image").exists() else []
    if code or len(images) != len(rows):
        raise RuntimeError(
            f"Generation stopped with {len(images)} images for {len(rows)} rows. See logs/generate.log."
        )
    return images


def label(out: Path):
    sdg = out / "sdg"
    code = run([
        "docker", "exec", ANOMALY, "python", "anomalygen/scripts/texture/pseudo_label.py",
        "--gen_root", str(sdg),
        "--output_dir", str(sdg / "pseudo_labels"),
        "--no_caption",
    ], out / "logs" / "labels.log", timeout=1800)
    coco = sdg / "pseudo_labels" / "coco_annotations.json"
    if code or not coco.exists():
        raise RuntimeError("Label export failed. See logs/labels.log.")
    return coco


def scene_name(scene, index):
    return f"{index:02d}_{scene['machine']}"


def main():
    import sys
    job = json.loads(Path(sys.argv[1]).read_text())
    out = ROOT / "results" / "studio" / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    plan = expand(job["settings"])
    (out / "plan.json").write_text(json.dumps({
        "prompt": job.get("prompt"),
        "summary": plan["summary"],
        "total": plan["total"],
        "held_total": plan["held_total"],
        "scene_total": plan["scene_total"],
        "held": [{k: cell[k] for k in ("machine_label", "damage_label", "position", "background", "reason")} for cell in plan["held"]],
    }, indent=2))
    say(out, "The settings are the plan. " + " ".join(plan["summary"]))
    reference = Path(job["reference"])
    if not reference.is_absolute():
        reference = ROOT / reference
    emit(out, kind="image", role="reference", path=str(reference.relative_to(ROOT)), caption="Reference photo for this run")
    if plan["held_total"]:
        say(out, f"{plan['held_total']} combinations are saved in the request and skipped, because those adapters are not in the checkpoint.")
    if not plan["generate"]:
        say(out, "Nothing in this grid has a trained adapter, so the run stops after recording the request.")
        emit(out, kind="done", generated=0, held=plan["held_total"], labels=None)
        return

    say(out, f"Generating {plan['total']} images from {plan['scene_total']} clean scenes. Checkpoint iter_000002000. No fine-tune.")
    editor = None
    try:
        ensure_container(AUGMENT, "augmentation")
        missing = [
            index for index, scene in enumerate(plan["scenes"], start=1)
            if not (out / "scenes" / f"{scene_name(scene, index)}.png").exists()
            or (out / "scenes" / f"{scene_name(scene, index)}.png").stat().st_size <= 10000
        ]
        if missing:
            editor = ensure_edit_server(out)
        else:
            say(out, f"All {plan['scene_total']} clean scenes are already saved. The image editor stays off.")
        dataset = out / "dataset"
        scene_paths = {}
        for index, scene in enumerate(plan["scenes"], start=1):
            destination = out / "scenes" / f"{scene_name(scene, index)}.png"
            source = reference if index == 1 else ROOT / scene["donor"]
            if index == 1:
                prompt = (
                    "Edit this real machinery photograph into one undamaged rental machine. "
                    f"Make it {scene['machine_label'].lower()} seen from a {scene['position']}. "
                    f"Place it in a {scene['background']}. Remove every person. "
                    "Intact glass, smooth panels, intact tires or tracks. "
                    "One machine only. Realistic inspection photo."
                )
            else:
                prompt = (
                    f"Edit this photograph of ONE {scene['machine_label'].lower()}. "
                    f"Change the viewpoint to a {scene['position']}. Replace the background with a {scene['background']}. "
                    "Keep one cab, one boom, and one bucket. The machine is parked and undamaged. "
                    "Realistic rental inspection photo."
                )
            config = out / "configs" / f"{destination.stem}.json"
            if destination.exists() and destination.stat().st_size > 10000:
                say(out, f"Clean scene {index} of {plan['scene_total']} is already saved. Keeping it.")
            else:
                say(out, f"Clean scene {index} of {plan['scene_total']}: {scene['machine_label']}, {scene['position']}, {scene['background']}.")
                write_edit_config(config, source, destination, prompt, 3000 + index)
                edit_image(out, config, destination)
            scene_paths[(scene["machine"], scene["position"], scene["background"])] = destination
            emit(out, kind="image", role="clean", path=str(destination.relative_to(ROOT)),
                 caption=f"{scene['machine_label']} · {scene['position']} · {scene['background']}")
            link_training_masks(dataset, scene["texture"], "unused")
            clean_dir = dataset / scene["texture"] / "clean_image"
            target = clean_dir / destination.name
            if target.exists():
                target.unlink()
            target.symlink_to(destination.resolve())

        say(out, "Stopping the image editor so AnomalyGen can use the GPU.")
        stop_edit_server(editor)
        editor = None
        ensure_anomaly()
        rows = []
        grouped = {}
        for cell in plan["generate"]:
            grouped.setdefault(cell["damage"], []).append(cell)
        damage_by_id = {item["id"]: item for item in DAMAGES}
        for damage_id, cells in grouped.items():
            damage = damage_by_id[damage_id]
            say(out, f"Placing {damage['label'].lower()} masks with Cosmos3-Nano and SAM 2.")
            # AMP pairs by clean-image stem. Keep only this damage's canvases visible by
            # pointing the dataset clean folder at the scenes this defect uses.
            texture = cells[0]["texture"]
            clean_dir = dataset / texture / "clean_image"
            for child in clean_dir.iterdir():
                if child.is_symlink() or child.is_file():
                    child.unlink()
            stems = set()
            for cell in cells:
                source = scene_paths[(cell["machine"], cell["position"], cell["background"])]
                link = clean_dir / source.name
                if source.name in stems:
                    continue
                stems.add(source.name)
                link.symlink_to(source.resolve())
            rows.extend(place_masks(out, dataset, damage, len(stems)))
            say(out, f"{damage['label']}: {len(stems)} masks placed.")
        say(out, "Generating damage inside those masks with the step-2000 checkpoint. UniPC, 35 steps.")
        images = generate(out, rows)
        for image in images:
            emit(out, kind="image", role="damaged", path=str(image.relative_to(ROOT)), caption=image.stem.replace("+", " "))
        coco = label(out)
        payload = json.loads(coco.read_text())
        say(out, f"Labels written. {len(payload.get('annotations', []))} COCO annotations in {coco.relative_to(ROOT)}.")
        emit(out, kind="done", generated=len(images), held=plan["held_total"], labels=str(coco.relative_to(ROOT)))
    except Exception as error:
        emit(out, kind="error", text=f"The pipeline stopped: {error}")
        raise
    finally:
        if editor is not None and editor.poll() is None:
            os.killpg(editor.pid, signal.SIGTERM)


if __name__ == "__main__":
    main()
