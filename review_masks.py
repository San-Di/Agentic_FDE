"""Render provisional training and target masks for visual review."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
annotations = json.loads((root / "results/bootstrap-annotations.json").read_text())
out = root / "results/mask-review"
out.mkdir(parents=True, exist_ok=True)
for name, item in annotations.items():
    for stage, field in (("clean", "target_polygons"), ("bootstrap", "bootstrap_polygons")):
        photo = Image.open(root / f"results/{stage}/{name}.png").convert("RGBA")
        overlay = Image.new("RGBA", photo.size)
        draw = ImageDraw.Draw(overlay)
        for points in item[field]:
            draw.polygon([tuple(p) for p in points], fill=(255, 40, 40, 60), outline=(255, 30, 30, 255), width=2)
        Image.alpha_composite(photo, overlay).convert("RGB").save(out / f"{name}-{stage}.jpg", quality=92)
print(f"Wrote mask overlays for {len(annotations)} reviewed cases")
