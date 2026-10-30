"""Paint damage inside a mask on a clean photo."""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import uuid
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "sunburst-compare"
REMOTE = os.environ.get("STUDIO_SSH_HOST", "")
SSH_KEY = os.environ.get("STUDIO_SSH_KEY", "")
SSH_BASE = [
    "ssh", "-4", "-i", SSH_KEY,
    "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", REMOTE,
]
SCP_BASE = [
    "scp", "-4", "-i", SSH_KEY,
    "-o", "ConnectTimeout=20", "-o", "BatchMode=yes",
]
REMOTE_ROOT = os.environ.get("STUDIO_REMOTE_ROOT", "")
REMOTE_WORK = f"{REMOTE_ROOT}/results/sunburst-compare/work"
CONTAINER = "cdf-route1-anomaly"
VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-6-astra")
IMAGE_MODEL = "gpt-image-2.5-sunburst"

PARTS = {
    "dent": "one metal body panel on the side of the machine, such as a door or the side of the house. Not the window, not a track or tire, and not the background.",
    "shattered_glass": "the cab window glass only.",
    "flat_tire": "one rubber tire, the whole tire from its top to where it meets the ground.",
}
PAINT = {
    "dent": "Add one photorealistic dent in the transparent area: a shallow crease in the metal with scuffed paint in the same body color and the same light. Leave every other part of the machine, the glass, the wheels or tracks, and the background unchanged.",
    "shattered_glass": "Shatter only the cab glass in the transparent area. Show an impact and radial cracks in the real window. Leave the frame, the body, the boom, the wheels or tracks, and the background unchanged.",
    "flat_tire": "Deflate the rubber tire in the transparent area so the bottom of that tire collapses against the ground. Keep the rim, the other tires, the body, and the background unchanged.",
}

CASES = [
    {"id": "reference-dent", "image": "results/studio/20261029-133546/reference.jpg", "kind": "dent", "seed": 9101,
     "original": "results/orchestration-compare/TrackedExcavator+dent_00000.png"},
    {"id": "reference-glass", "image": "results/studio/20261029-133546/reference.jpg", "kind": "shattered_glass", "seed": 9102,
     "original": "results/orchestration-compare/TrackedExcavator+shattered_glass_00000.png"},
    {"id": "reference-tire", "image": "results/studio/20261029-133546/reference.jpg", "kind": "flat_tire", "seed": 9103,
     "original": "results/orchestration-compare/WheeledExcavator+flat_tire_00000.png"},
    {"id": "tracked-side-dent", "image": "results/studio/20261029-133546/scenes/01_tracked.png", "kind": "dent", "seed": 9104},
    {"id": "tracked-site-glass", "image": "results/studio/20261029-133546/scenes/05_tracked.png", "kind": "shattered_glass", "seed": 9105},
    {"id": "wheeled-tire", "image": "results/studio/20261029-133546/scenes/14_wheeled.png", "kind": "flat_tire", "seed": 9106},
]


def on_gpu_host():
    return Path(REMOTE_ROOT).exists() and Path("/usr/bin/docker").exists()


def api(path, payload=None, raw=None, content_type="application/json", timeout=180):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set. The Sunburst comparison needs it for the part box and the paint step.")
    request = urllib.request.Request(
        "https://api.openai.com" + path,
        data=raw if raw is not None else json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:800]
        raise RuntimeError(f"OpenAI {path} returned {error.code}: {detail}") from error


def locate_part(image: Image.Image, kind: str):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    width, height = image.size
    prompt = (
        "Locate one part on this construction-machine photo. "
        f"The image is {width} pixels wide and {height} pixels tall. "
        "Reply with JSON only: {\"found\": true, \"x0\": left, \"y0\": top, \"x1\": right, \"y1\": bottom}. "
        "The box is in pixels and should include a small margin around the part. "
        f"If it is not visible, reply {{\"found\": false}}. Part: {PARTS[kind]}"
    )
    body = api("/v1/chat/completions", {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
            ],
        }],
        "response_format": {"type": "json_object"},
    }, timeout=90)
    text = body["choices"][0]["message"]["content"]
    found = json.loads(text)
    if not found.get("found"):
        raise RuntimeError(f"No visible part for {kind}.")
    box = [int(found["x0"]), int(found["y0"]), int(found["x1"]), int(found["y1"])]
    box[0] = max(0, min(width - 2, box[0]))
    box[1] = max(0, min(height - 2, box[1]))
    box[2] = max(box[0] + 2, min(width, box[2]))
    box[3] = max(box[1] + 2, min(height, box[3]))
    return box


def run(command, timeout=600):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-800:])
    return result.stdout


def cut_mask(image_path: Path, case_id: str, box, kind: str, seed: int, mask_path: Path):
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if on_gpu_host():
        work = OUT / "work"
        work.mkdir(parents=True, exist_ok=True)
        command = [
            "docker", "exec", CONTAINER, "python", str(ROOT / "shape_on_part.py"),
            "--image", str(image_path), "--box", *map(str, box),
            "--kind", kind, "--seed", str(seed),
            "--part", str(work / f"{case_id}-part.png"),
            "--mask", str(mask_path),
        ]
        print(run(command), flush=True)
        return
    run(SSH_BASE + ["mkdir", "-p", REMOTE_WORK])
    run(SCP_BASE + [str(ROOT / "shape_on_part.py"), str(image_path), f"{REMOTE}:{REMOTE_WORK}/"])
    remote_image = f"{REMOTE_WORK}/{image_path.name}"
    remote_mask = f"{REMOTE_WORK}/{case_id}-mask.png"
    remote_part = f"{REMOTE_WORK}/{case_id}-part.png"
    print(run(SSH_BASE + [
        "docker", "exec", CONTAINER, "python", f"{REMOTE_WORK}/shape_on_part.py",
        "--image", remote_image, "--box", *map(str, box),
        "--kind", kind, "--seed", str(seed),
        "--part", remote_part, "--mask", remote_mask,
    ]), flush=True)
    run(SCP_BASE + [f"{REMOTE}:{remote_mask}", str(mask_path)])


def multipart(fields, files):
    boundary = "----cdf" + uuid.uuid4().hex
    chunks = []
    for name, value in fields:
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    for name, filename, content_type, data in files:
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode()
            + data + b"\r\n"
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), "multipart/form-data; boundary=" + boundary


def edit_mask_bytes(damage: Image.Image):
    # Transparent pixels are the area Sunburst may change.
    alpha = damage.point(lambda p: 0 if p >= 128 else 255)
    rgba = Image.new("RGBA", damage.size, (0, 0, 0, 255))
    rgba.putalpha(alpha)
    buffer = io.BytesIO()
    rgba.save(buffer, format="PNG")
    return buffer.getvalue()


def paint(clean: Image.Image, damage: Image.Image, kind: str):
    image_bytes = io.BytesIO()
    clean.save(image_bytes, format="PNG")
    body, content_type = multipart(
        [("model", IMAGE_MODEL), ("prompt", PAINT[kind]), ("quality", "high"), ("size", "auto")],
        [
            ("image[]", "clean.png", "image/png", image_bytes.getvalue()),
            ("mask", "mask.png", "image/png", edit_mask_bytes(damage)),
        ],
    )
    result = api("/v1/images/edits", raw=body, content_type=content_type, timeout=300)
    edited = Image.open(io.BytesIO(base64.b64decode(result["data"][0]["b64_json"]))).convert("RGB")
    edited = edited.resize(clean.size, Image.Resampling.LANCZOS)
    painted = clean.copy()
    painted.paste(edited, mask=damage)
    return edited, painted


def thumb(image, width):
    height = max(1, int(image.height * width / image.width))
    return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def overlay(clean, damage):
    base = clean.convert("RGBA")
    red = Image.new("RGBA", clean.size, (220, 40, 40, 0))
    alpha = damage.point(lambda p: 110 if p >= 128 else 0)
    red.putalpha(alpha)
    return Image.alpha_composite(base, red).convert("RGB")


def sheet(rows):
    width = 280
    tiles = []
    for row in rows:
        cells = [thumb(image, width) for image in row["images"]]
        tiles.append(cells)
    label_h = 28
    row_h = max(cell.height for cells in tiles for cell in cells) + label_h
    canvas = Image.new("RGB", (width * 4 + 16, row_h * len(tiles) + 8), (24, 26, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for r, (row, cells) in enumerate(zip(rows, tiles)):
        draw.text((8, r * row_h + 6), row["label"], fill=(240, 220, 80), font=font)
        for c, cell in enumerate(cells):
            canvas.paste(cell, (8 + c * width, r * row_h + label_h))
    canvas.save(OUT / "comparison.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = []
    for case in CASES:
        source = ROOT / case["image"]
        clean = Image.open(source).convert("RGB")
        mask_path = OUT / "masks" / f"{case['id']}.png"
        painted_path = OUT / "painted" / f"{case['id']}.png"
        print(f"{case['id']}: locating {case['kind']}", flush=True)
        try:
            box = locate_part(clean, case["kind"])
            print(f"{case['id']}: box {box}", flush=True)
            cut_mask(source, case["id"], box, case["kind"], case["seed"], mask_path)
            damage = Image.open(mask_path).convert("L")
            raw, painted = paint(clean, damage, case["kind"])
            painted_path.parent.mkdir(parents=True, exist_ok=True)
            (OUT / "raw").mkdir(parents=True, exist_ok=True)
            painted.save(painted_path)
            raw.save(OUT / "raw" / f"{case['id']}.png")
            original_path = ROOT / case["original"] if case.get("original") else None
            original = Image.open(original_path).convert("RGB") if original_path and original_path.exists() else Image.new("RGB", clean.size, (20, 20, 20))
            rows.append({
                "label": case["id"],
                "images": [original, clean, overlay(clean, damage), painted],
            })
            manifest.append({"id": case["id"], "kind": case["kind"], "box": box, "painted": str(painted_path.relative_to(ROOT)), "mask": str(mask_path.relative_to(ROOT))})
            print(f"{case['id']}: painted", flush=True)
        except Exception as error:
            manifest.append({"id": case["id"], "error": str(error)})
            print(f"{case['id']}: {error}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if rows:
        sheet(rows)
        print(f"Wrote {OUT / 'comparison.png'}", flush=True)
    failed = [item for item in manifest if "error" in item]
    if failed:
        raise SystemExit(f"{len(failed)} case(s) failed. See {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
