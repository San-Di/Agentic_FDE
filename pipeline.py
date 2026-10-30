"""Build an AnomalyGen dataset from a business request and reference photos."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
MAX_SIDE = 1280


def load_request(path):
    request = json.loads(path.read_text())
    required = ("name", "business", "classes", "scenes", "examples", "annotations")
    missing = [key for key in required if key not in request]
    if missing:
        raise SystemExit(f"Request is missing {missing}")
    return request


def class_key(texture, defect):
    return texture + "+" + defect


def resize_max(image, max_side):
    image = image.convert("RGB")
    if max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def binary_mask(size, polygons):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError("A defect mask needs at least three points")
        draw.polygon([tuple(point) for point in polygon], fill=255)
    low, high = mask.getextrema()
    if low == high:
        raise ValueError(f"Mask on {size} is empty or solid")
    return mask


def scaled_polygons(polygons, size):
    width, height = size
    scaled = []
    for polygon in polygons:
        points = []
        for x, y in polygon:
            if not 0 <= x <= 1 or not 0 <= y <= 1:
                raise ValueError(f"Polygon point {(x, y)} is not normalized")
            points.append((x * (width - 1), y * (height - 1)))
        scaled.append(points)
    return scaled


def prepare(request):
    dataset = root / "dataset_ft"
    if dataset.exists():
        shutil.rmtree(dataset)
    review = json.loads((root / request["annotations"]).read_text())
    declared = {class_key(item["texture"], item["defect"]) for item in request["classes"]}
    previews = root / "results" / request["name"] / "mask-preview"
    previews.mkdir(parents=True, exist_ok=True)
    provenance = []

    for example in request["examples"]:
        key = class_key(example["texture"], example["defect"])
        if key not in declared:
            raise SystemExit(f"Example {example['id']} is not in classes")
        image_path = root / example["file"]
        if not image_path.exists():
            raise SystemExit(f"Missing reference photo {image_path}")
        destination = dataset / example["texture"] / "anomaly_image" / example["defect"]
        mask_dir = dataset / example["texture"] / "mask" / example["defect"]
        destination.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        if example.get("mask"):
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(root / example["mask"]).convert("L")
            if image.size != mask.size:
                raise SystemExit(f"Mask size does not match {example['id']}")
            mask = mask.point(lambda value: 255 if value >= 128 else 0)
        else:
            image = resize_max(Image.open(image_path), MAX_SIDE)
            mask = binary_mask(image.size, scaled_polygons([example["polygons"]], image.size))
        if mask.getextrema()[0] == mask.getextrema()[1]:
            raise SystemExit(f"Mask for {example['id']} has no defect region")
        image.save(destination / f"{example['id']}.png")
        mask.save(mask_dir / f"{example['id']}_mask.png")
        overlay = image.convert("RGB")
        red = Image.new("RGB", image.size, (255, 40, 40))
        overlay.paste(red, mask=mask.point(lambda value: 110 if value else 0))
        overlay.save(previews / f"{example['id']}.jpg", quality=80)
        provenance.append({
            "id": example["id"], "class": key, "source": example.get("source"),
            "note": example.get("note"), "pixels": list(image.size),
        })

    records, validation = [], []
    seen = set()
    for index, scene in enumerate(request["scenes"]):
        key = class_key(scene["texture"], scene["defect"])
        if key not in declared:
            raise SystemExit(f"Scene {scene['id']} is not in classes")
        entry = review[scene["id"]]
        if entry.get("clean_accepted") is not True:
            raise SystemExit(f"Clean scene {scene['id']} was not accepted")
        clean_path = root / scene["clean"]
        clean = Image.open(clean_path).convert("RGB")
        clean_dir = dataset / scene["texture"] / "clean_image"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_dest = clean_dir / f"{scene['id']}.png"
        shutil.copy2(clean_path, clean_dest)
        target_dir = root / "results" / request["name"] / "target-masks"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{scene['id']}_target.png"
        binary_mask(clean.size, entry["target_polygons"]).save(target)
        record = {
            "image_filename": str(clean_dest), "mask_filename": str(target),
            "anomaly_type": key, "num_generated_images": 1,
            "guidance": 3.0, "num_steps": 35, "crop_ratio": 2.0,
            "seed": 6100 + index, "index": index,
        }
        records.append(record)
        if key not in seen:
            seen.add(key)
            validation.append({**record, "num_generated_images": 1, "seed": 7100 + index, "index": len(validation)})

    missing = declared - seen
    if missing:
        raise SystemExit(f"No clean scene for {sorted(missing)}")
    covered = {scene["texture"] for scene in request["scenes"]}
    unmet = [name for name in request.get("requested_equipment", []) if name not in covered]

    run_dir = root / "results" / request["name"]
    for path, rows in (
        (run_dir / "target-masks" / "testcase.jsonl", records),
        (run_dir / "validation" / "testcase.jsonl", validation),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    spec_path = dataset / "defect_spec.jsonl"
    spec_path.write_text("".join(json.dumps({
        "defect_type": class_key(item["texture"], item["defect"]),
        "spatial_dependency": item["spatial_dependency"],
        "roi_prompt_defect_location": item["roi_prompt"],
    }) + "\n" for item in request["classes"]))
    recipe = {
        "task_type": "texture_ft", "experiment": "anomalygen_texture_ft",
        "job_name": "anomalygen_texture_ft", "dataset_name": request["name"],
        "dataset_path": str(dataset),
        "anomaly_types": [[item["texture"], item["defect"]] for item in request["classes"]],
        "testcase_jsonl": str(run_dir / "validation" / "testcase.jsonl"),
        "model_size": "nano", "max_iter": request["iterations"],
        "cycle_lengths": request["iterations"], "warm_up_steps": 200,
        "save_iter": 2000, "validation_iter": 2000, "run_validation_on_start": True,
        "validation_batch_size": 1, "max_sequence_length": 2048,
        "input_size_ladder": [256, 512], "activation_checkpointing_mode": "full",
        "logging_iter": 10, "num_workers": 2, "seed": 42,
    }
    recipe_path = root / "configs" / f"train_{request['name']}.yaml"
    recipe_path.write_text(json.dumps(recipe, indent=2))
    undamaged = run_dir / "export" / "undamaged"
    undamaged.mkdir(parents=True, exist_ok=True)
    for scene in request["scenes"]:
        shutil.copy2(dataset / scene["texture"] / "clean_image" / f"{scene['id']}.png", undamaged / f"{scene['id']}.png")
    manifest = {
        "business": request["business"],
        "reference_photo": request["reference_photo"],
        "dataset": str(dataset),
        "recipe": str(recipe_path),
        "generate_jsonl": str(run_dir / "target-masks" / "testcase.jsonl"),
        "training_examples": provenance,
        "generation_scenes": [scene["id"] for scene in request["scenes"]],
        "unmet_equipment": unmet,
        "unmet_reason": (
            "No undamaged photo was supplied for this equipment, so the pipeline does not invent one. "
            "Related real defect photos are still used to teach a damage class that will be painted onto the equipment that does have a clean photo."
            if unmet else ""
        ),
        "held_out_references": request.get("held_out_references", []),
        "training_budget": request["iterations"],
        "generator": "CDF AnomalyGen texture fine-tune. Defect pixels come from the reference photos, not from an image-edit model.",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "examples": len(provenance), "scenes": len(records),
        "unmet_equipment": unmet, "recipe": str(recipe_path),
    }, indent=2))
    return manifest


def export_generated(request):
    run_dir = root / "results" / request["name"]
    searched = run_dir / "searched"
    if not (searched / "reconstructed_image").exists():
        searched = run_dir / "sdg"
    damaged = run_dir / "export" / "damaged"
    labels = run_dir / "export" / "labels"
    damaged.mkdir(parents=True, exist_ok=True)
    if (searched / "pseudo_labels").exists():
        shutil.copytree(searched / "pseudo_labels", labels, dirs_exist_ok=True)
    images = searched / "reconstructed_image"
    if images.exists():
        for path in images.glob("*.png"):
            shutil.copy2(path, damaged / path.name)
    print(f"Exported {len(list(damaged.glob('*.png')))} generated images to {damaged}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, default=root / "business.json")
    parser.add_argument("--run", action="store_true", help="Fine-tune AnomalyGen and generate the dataset")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    request = load_request(args.request)
    if args.export_only:
        export_generated(request)
        return
    prepare(request)
    if not args.run:
        print("Dataset prepared. Pass --run to fine-tune and generate.")
        return
    command = [
        sys.executable, str(root / "run_anomalygen.py"),
        "--run-name", request["name"],
        "--recipe", str(root / "configs" / f"train_{request['name']}.yaml"),
        "--dataset-dir", str(root / "dataset_ft"),
        "--generate-jsonl", str(root / "results" / request["name"] / "target-masks" / "testcase.jsonl"),
        "--output-root", str(root / "results" / request["name"]),
    ]
    subprocess.run(command, check=True)
    export_generated(request)


if __name__ == "__main__":
    main()
