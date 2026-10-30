"""Clean scenes, a mask on each one, then paint."""
from __future__ import annotations

import base64
import io
import json
import os
import signal
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

from studio_catalog import expand
from studio_run import (
    ANOMALY,
    ROOT,
    edit_image,
    emit,
    ensure_container,
    ensure_anomaly,
    ensure_edit_server,
    say,
    scene_name,
    stop_edit_server,
    write_edit_config,
)

SECRETS = Path("/home/ubuntu/cdf-orchestration/secrets.env")
TOKEN_FACTORY = "https://api.tokenfactory.nebius.com"
NEMOTRON_VL = "nvidia/Nemotron-3.5-Super-VL"
VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-6-astra")
IMAGE_MODEL = "gpt-image-2.5-sunburst"
PARTS = {
    "dent": "one metal body panel on the side of the machine, such as a door or the side of the house. Not the window, not a track or tire, and not the background.",
    "shattered_glass": "the cab window glass only.",
    "flat_tire": "one rubber tire, the whole tire from its top to where it meets the ground.",
    "bucket_dent": "the digging bucket, including its teeth. Not the boom and not the ground.",
}
PAINT = {
    "dent": "Add one photorealistic dent in the transparent area: a shallow crease in the metal with scuffed paint in the same body color and the same light. Leave every other part of the machine and the background unchanged.",
    "shattered_glass": "Shatter only the cab glass in the transparent area. Show an impact and radial cracks in the real window. Leave the frame, the body, and the background unchanged.",
    "flat_tire": "Deflate the rubber tire in the transparent area so the bottom of that tire collapses against the ground. Keep the rim, the other tires, and the background unchanged.",
    "bucket_dent": "Add one photorealistic dent or bent tooth on the bucket in the transparent area. Leave the boom and the ground unchanged.",
}


def secret_value(name):
    if SECRETS.exists():
        for line in SECRETS.read_text().splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            key = key.replace("export ", "").strip()
            if key == name:
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    return os.environ.get(name, "").strip()


def openai_key():
    value = secret_value("OPENAI_API_KEY")
    if value:
        return value
    raise RuntimeError("OPENAI_API_KEY is missing from the orchestration secrets file.")


def vision_target():
    # Nebius is used only when RUN_TARGET=nebius.
    if os.environ.get("RUN_TARGET", "aws").strip().lower() == "nebius":
        key = secret_value("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError("RUN_TARGET=nebius but NEBIUS_API_KEY is missing.")
        return TOKEN_FACTORY, key, os.environ.get("NEBIUS_VISION_MODEL", NEMOTRON_VL)
    return "https://api.openai.com", openai_key(), VISION_MODEL


def api(base, key, path, payload=None, raw=None, content_type="application/json", timeout=180):
    request = urllib.request.Request(
        base + path,
        data=raw if raw is not None else json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise RuntimeError(f"{base} {path} returned {error.code}: {detail}") from error


def locate_part(base, key, model, image: Image.Image, kind: str):
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
    body = api(base, key, "/v1/chat/completions", {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
            ],
        }],
        "response_format": {"type": "json_object"},
    }, timeout=90)
    found = json.loads(body["choices"][0]["message"]["content"])
    if not found.get("found"):
        raise RuntimeError(f"No visible part for {kind}.")
    box = [int(found["x0"]), int(found["y0"]), int(found["x1"]), int(found["y1"])]
    box[0] = max(0, min(width - 2, box[0]))
    box[1] = max(0, min(height - 2, box[1]))
    box[2] = max(box[0] + 2, min(width, box[2]))
    box[3] = max(box[1] + 2, min(height, box[3]))
    return box


def cut_mask(image_path: Path, box, kind: str, seed: int, mask_path: Path):
    from studio_run import run
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    code = run([
        "docker", "exec", ANOMALY, "python", str(ROOT / "shape_on_part.py"),
        "--image", str(image_path), "--box", *map(str, box),
        "--kind", kind, "--seed", str(seed),
        "--part", str(mask_path.with_name(mask_path.stem + "-part.png")),
        "--mask", str(mask_path),
    ], mask_path.with_suffix(".log"), timeout=600)
    if code or not mask_path.exists():
        raise RuntimeError(f"SAM did not write a mask for {image_path.name}.")


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


def paint(key, clean: Image.Image, damage: Image.Image, kind: str):
    image_bytes = io.BytesIO()
    clean.save(image_bytes, format="PNG")
    alpha = damage.point(lambda pixel: 0 if pixel >= 128 else 255)
    rgba = Image.new("RGBA", damage.size, (0, 0, 0, 255))
    rgba.putalpha(alpha)
    mask_bytes = io.BytesIO()
    rgba.save(mask_bytes, format="PNG")
    body, content_type = multipart(
        [("model", IMAGE_MODEL), ("prompt", PAINT[kind]), ("quality", "high"), ("size", "auto")],
        [
            ("image[]", "clean.png", "image/png", image_bytes.getvalue()),
            ("mask", "mask.png", "image/png", mask_bytes.getvalue()),
        ],
    )
    result = api("https://api.openai.com", key, "/v1/images/edits", raw=body, content_type=content_type, timeout=300)
    edited = Image.open(io.BytesIO(base64.b64decode(result["data"][0]["b64_json"]))).convert("RGB")
    edited = edited.resize(clean.size, Image.Resampling.LANCZOS)
    painted = clean.copy()
    painted.paste(edited, mask=damage)
    return painted


def clean_with_gpt(key, source: Path, destination: Path, prompt: str):
    mime = "image/png" if source.suffix.lower() == ".png" else "image/jpeg"
    body, content_type = multipart(
        [("model", IMAGE_MODEL), ("prompt", prompt), ("quality", "high"), ("size", "auto")],
        [("image[]", source.name, mime, source.read_bytes())],
    )
    result = api("https://api.openai.com", key, "/v1/images/edits", raw=body, content_type=content_type, timeout=300)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(result["data"][0]["b64_json"]))


def clean_prompt(scene, first):
    if first:
        return (
            "Edit this real machinery photograph into one undamaged rental machine. "
            f"Make it one {scene['machine_label'].lower()} seen from a {scene['position']}. "
            f"Place it in a {scene['background']}. Remove every person. "
            "Intact glass, smooth panels, intact tires or tracks. "
            "One machine only. Realistic inspection photo."
        )
    return (
        f"Edit this photograph into ONE undamaged {scene['machine_label'].lower()}. "
        f"Change the viewpoint to a {scene['position']}. Replace the background with a {scene['background']}. "
        "Keep one cab. The machine is parked and undamaged. Realistic rental inspection photo."
    )


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    settings = dict(job.get("settings") or {})
    pipeline = settings.get("pipeline") if settings.get("pipeline") in {"sunburst", "gptclean"} else "sunburst"
    settings["pipeline"] = pipeline
    gpt_clean = pipeline == "gptclean"
    out = ROOT / "results" / "studio" / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    plan = expand(settings)
    (out / "plan.json").write_text(json.dumps({
        "pipeline": pipeline,
        "prompt": job.get("prompt"),
        "summary": plan["summary"],
        "total": plan["total"],
        "scene_total": plan["scene_total"],
    }, indent=2))
    say(out, ("GPT clean plan. " if gpt_clean else "Sunburst plan. ") + " ".join(plan["summary"]))
    reference = Path(job["reference"])
    if not reference.is_absolute():
        reference = ROOT / reference
    emit(out, kind="image", role="reference", path=str(reference.relative_to(ROOT)), caption="Reference photo for this run")
    if not plan["generate"]:
        say(out, "Nothing in this grid applies to the checked machines, so the run stops.")
        emit(out, kind="done", generated=0, held=0, labels=None, results=[])
        return

    say(out, f"Generating {plan['scene_total']} clean scenes, then {plan['total']} labeled damage photos.")
    editor = None
    labels = []
    try:
        vision_base, vision_key, vision_model = vision_target()
        paint_key = openai_key()
        say(out, f"Part boxes use {vision_model}. Paint uses {IMAGE_MODEL}.")
        missing = [
            index for index, scene in enumerate(plan["scenes"], start=1)
            if not (out / "scenes" / f"{scene_name(scene, index)}.png").exists()
            or (out / "scenes" / f"{scene_name(scene, index)}.png").stat().st_size <= 10000
        ]
        editor = None
        if missing and not gpt_clean:
            ensure_container("cdf-route1-augmentation", "augmentation")
            editor = ensure_edit_server(out)
        elif not missing:
            say(out, f"All {plan['scene_total']} clean scenes are already saved.")
        scene_paths = {}
        for index, scene in enumerate(plan["scenes"], start=1):
            destination = out / "scenes" / f"{scene_name(scene, index)}.png"
            config = out / "configs" / f"{destination.stem}.json"
            if destination.exists() and destination.stat().st_size > 10000:
                say(out, f"Clean scene {index} of {plan['scene_total']} is already saved. Keeping it.")
            else:
                say(out, f"Clean scene {index} of {plan['scene_total']}: {scene['machine_label']}, {scene['position']}, {scene['background']}.")
                prompt = clean_prompt(scene, index == 1)
                if gpt_clean:
                    clean_with_gpt(paint_key, reference, destination, prompt)
                else:
                    write_edit_config(config, reference, destination, prompt, 3000 + index)
                    edit_image(out, config, destination)
            scene_paths[(scene["machine"], scene["position"], scene["background"])] = destination
            emit(out, kind="image", role="clean", path=str(destination.relative_to(ROOT)),
                 caption=f"{scene['machine_label']} · {scene['position']} · {scene['background']}")

        say(out, "Freeing the GPU so SAM 2 can cut the masks.")
        stop_edit_server(editor)
        editor = None
        ensure_anomaly()
        for index, cell in enumerate(plan["generate"], start=1):
            caption = f"{cell['machine_label']} · {cell['damage_label']} · {cell['position']} · {cell['background']}"
            say(out, f"Damage {index} of {plan['total']}: {caption}.")
            clean_path = scene_paths[(cell["machine"], cell["position"], cell["background"])]
            stem = f"{index:02d}_{cell['machine']}_{cell['damage']}"
            mask_path = out / "masks" / f"{stem}.png"
            painted_path = out / "damaged" / f"{stem}.png"
            try:
                clean = Image.open(clean_path).convert("RGB")
                box = locate_part(vision_base, vision_key, vision_model, clean, cell["damage"])
                cut_mask(clean_path, box, cell["damage"], 9100 + index, mask_path)
                damage = Image.open(mask_path).convert("L")
                painted = paint(paint_key, clean, damage, cell["damage"])
                painted_path.parent.mkdir(parents=True, exist_ok=True)
                painted.save(painted_path)
            except Exception as error:
                say(out, f"Skipped {caption}: {error}")
                continue
            record = {
                "image": str(painted_path.relative_to(ROOT)),
                "mask": str(mask_path.relative_to(ROOT)),
                "clean": str(clean_path.relative_to(ROOT)),
                "machine": cell["machine_label"],
                "damage": cell["damage_label"],
                "position": cell["position"],
                "background": cell["background"],
            }
            labels.append(record)
            emit(out, kind="result", caption=caption, damaged=record["image"], **record)
        label_path = out / "labels.json"
        label_path.write_text(json.dumps({"images": labels}, indent=2))
        say(out, f"Labels written. {len(labels)} photos in {label_path.relative_to(ROOT)}.")
        emit(out, kind="done", generated=len(labels), held=0, labels=str(label_path.relative_to(ROOT)), results=labels)
    except Exception as error:
        emit(out, kind="error", text=f"The pipeline stopped: {error}")
        raise
    finally:
        if editor is not None and editor.poll() is None:
            os.killpg(editor.pid, signal.SIGTERM)


if __name__ == "__main__":
    main()
