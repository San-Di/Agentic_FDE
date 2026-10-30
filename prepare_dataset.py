"""Build AnomalyGen dataset records from the reviewed annotations."""
import json
from pathlib import Path
import shutil
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
plan = json.loads((root / "results/planning/reviewed-plan.json").read_text())
review = json.loads((root / "results/bootstrap-annotations.json").read_text())
generation = root / "results/target-masks"
validation = root / "results/validation"
generation.mkdir(parents=True, exist_ok=True)
validation.mkdir(parents=True, exist_ok=True)
records, provenance, val_records = [], [], []
seen = set()

def mask_for(image, polygons, destination):
    mask = Image.new("L", image.size)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        for x, y in polygon:
            if not (0 <= x < image.width and 0 <= y < image.height):
                raise ValueError(f"Mask point {(x, y)} is outside {image.size}")
        draw.polygon([tuple(p) for p in polygon], fill=255)
    if mask.getbbox() is None:
        raise ValueError(f"Empty mask for {destination}")
    mask.save(destination)

for i, variant in enumerate(plan["variants"]):
    name, defect, texture = variant["id"], variant["defect"], variant["texture"]
    entry = review[name]
    if entry.get("clean_accepted") is not True:
        raise ValueError(f"Review did not accept the clean input for {name}")
    use_bootstrap = entry.get("bootstrap_accepted") is True
    real_reference = root / "dataset" / texture / "anomaly_image" / defect / "real_reference_crop.png"
    real_mask = root / "dataset" / texture / "mask" / defect / "real_reference_crop_mask.png"
    if not use_bootstrap and not (
        entry.get("use_real_reference_only") is True and real_reference.exists() and real_mask.exists()
    ):
        raise ValueError(f"No accepted training source for {name}")
    base = root / "dataset" / texture
    for sub in ("clean_image", f"anomaly_image/{defect}", f"mask/{defect}"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    clean_path = root / f"results/clean/{name}.png"
    defect_path = root / f"results/bootstrap/{name}.png"
    clean = Image.open(clean_path).convert("RGB")
    clean_dest = base / "clean_image" / f"{name}.png"
    shutil.copy2(clean_path, clean_dest)
    synthetic_image = base / "anomaly_image" / defect / f"synthetic_{name}.png"
    synthetic_mask = base / "mask" / defect / f"synthetic_{name}_mask.png"
    if use_bootstrap:
        damaged = Image.open(defect_path).convert("RGB")
        shutil.copy2(defect_path, synthetic_image)
        mask_for(damaged, entry["bootstrap_polygons"], synthetic_mask)
    else:
        # Never leave an older, subsequently rejected bootstrap in the training tree.
        synthetic_image.unlink(missing_ok=True)
        synthetic_mask.unlink(missing_ok=True)
    target = generation / f"{name}_target.png"
    mask_for(clean, entry["target_polygons"], target)
    record = {"image_filename": str(clean_dest), "mask_filename": str(target),
        "anomaly_type": texture + "+" + defect, "num_generated_images": 1,
        "guidance": 3.0, "num_steps": 35, "crop_ratio": 2.0, "seed": 4100 + i, "index": i}
    records.append(record)
    if defect not in seen:
        seen.add(defect)
        val_records.append({**record, "seed": 5100 + i, "index": len(val_records)})
    provenance.append({"id": name, "training_image": str(defect_path if use_bootstrap else real_reference),
        "image_provenance": "Qwen synthetic bootstrap" if use_bootstrap else "user photo crop; rejected bootstrap excluded",
        "annotation_provenance": "agent-reviewed polygons",
        "target_mask_provenance": "agent-selected edit region; not measured defect ground truth",
        "review": entry})
for path, rows in ((generation / "testcase.jsonl", records), (validation / "testcase.jsonl", val_records)):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
recipe = {
    "task_type": "texture_ft", "experiment": "anomalygen_texture_ft", "job_name": "anomalygen_texture_ft",
    "dataset_name": "excavator_route1", "dataset_path": str(root / "dataset"),
    "anomaly_types": [["TrackedExcavator", "dent"], ["TrackedExcavator", "shattered_glass"], ["WheeledExcavator", "flat_tire"]],
    "testcase_jsonl": str(validation / "testcase.jsonl"), "model_size": "nano",
    "max_iter": 600, "cycle_lengths": 600, "warm_up_steps": 50,
    "save_iter": 300, "validation_iter": 300, "run_validation_on_start": True,
    "validation_batch_size": 1, "max_sequence_length": 2048,
    "input_size_ladder": [256, 512], "activation_checkpointing_mode": "full",
    "logging_iter": 10, "num_workers": 2, "seed": 42,
}
(root / "configs/train.yaml").write_text(json.dumps(recipe, indent=2))
(root / "results/dataset-provenance.json").write_text(json.dumps({
    "examples": provenance,
    "mask_stage": "Agent-reviewed explicit masks in CDF's testcase format; not a claim that AMG ran",
    "validation_limitation": "Same six clean bases; three cases select a checkpoint, not an independent real-world test set",
    "training_budget": "600-iteration pilot, shorter than NVIDIA's 15000-iteration default",
}, indent=2))
print("Prepared", len(records), "generation cases and", len(val_records), "internal validation cases")
