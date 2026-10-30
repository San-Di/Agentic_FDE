"""Check actual output/label geometry and changes outside the requested edit region."""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
mapping = json.loads((root / "results/anomaly-output-map.json").read_text())
rows = []
for name, relative in mapping.items():
    output = Image.open(root / relative).convert("RGB")
    clean = Image.open(root / f"results/clean/{name}.png").convert("RGB")
    mask = Image.open(root / f"results/target-masks/{name}_target.png").convert("L")
    if output.size != clean.size or mask.size != clean.size:
        raise ValueError(f"Size mismatch for {name}: {output.size}, {clean.size}, {mask.size}")
    delta = np.abs(np.asarray(output, dtype=np.float32) - np.asarray(clean, dtype=np.float32))
    outside = np.asarray(mask) == 0
    rows.append({"id": name, "output": relative, "size": list(output.size),
        "target_fraction": float(1 - outside.mean()),
        "outside_target_mean_absolute_difference_0_to_255": float(delta[outside].mean()),
        "inside_target_mean_absolute_difference_0_to_255": float(delta[~outside].mean()),
        "note": "Pixel changes do not establish visible damage or label accuracy."})
coco_path = root / "results/anomalygen/searched/pseudo_labels/coco_annotations.json"
coco = json.loads(coco_path.read_text())
images = {i["id"]: i for i in coco["images"]}
for annotation in coco["annotations"]:
    info = images[annotation["image_id"]]
    x, y, width, height = annotation["bbox"]
    if not all(math.isfinite(v) for v in (x, y, width, height)) or not (
        x >= 0 and y >= 0 and width > 0 and height > 0
        and x + width <= info["width"] + 1 and y + height <= info["height"] + 1
    ):
        raise ValueError(f"Invalid bounding box: {annotation}")
report = {"generated_cases": rows, "coco_images": len(images),
    "coco_annotations": len(coco["annotations"]), "categories": coco["categories"],
    "label_status": "Provisional labels derived from generation masks; visual acceptance is separate."}
(root / "results/output-audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
