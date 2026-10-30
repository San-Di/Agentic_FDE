"""Polygons for the two visible defects in the reference photo."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
path = root / "inputs/reference.jpg"
source = Image.open(path).convert("RGB")
annotations = [
    {"class": "shattered_glass", "crop": [305, 5, 510, 351], "polygons": [
        [[347, 63], [435, 34], [440, 210], [331, 224]],
        [[450, 36], [478, 27], [488, 145], [466, 220], [444, 218]],
        [[330, 249], [428, 242], [421, 325], [338, 331]],
    ]},
    {"class": "dent", "crop": [470, 320, 826, 452], "polygons": [
        [[503, 355], [545, 332], [597, 344], [643, 348], [683, 337], [768, 350],
         [811, 375], [795, 406], [707, 434], [515, 420], [483, 393]],
    ]},
]
records = []
for item in annotations:
    defect = item["class"]
    base = root / "dataset/TrackedExcavator"
    for folder in (base / "anomaly_image" / defect, base / "mask" / defect):
        folder.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", source.size)
    draw = ImageDraw.Draw(mask)
    for polygon in item["polygons"]:
        draw.polygon([tuple(p) for p in polygon], fill=255)
    source.crop(item["crop"]).save(base / "anomaly_image" / defect / "real_reference_crop.png")
    mask.crop(item["crop"]).save(base / "mask" / defect / "real_reference_crop_mask.png")
    records.append({**item, "image_provenance": "one user-supplied real photograph",
        "annotation_provenance": "agent-drawn provisional polygon", "independent_holdout": False})
(root / "results/reference-annotations.json").write_text(json.dumps({
    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "annotations": records,
    "flat_tire_reference": None, "warning": "All crops share the same source photograph; no real test split exists.",
}, indent=2))
print("Prepared two real-reference crops and provisional masks.")
