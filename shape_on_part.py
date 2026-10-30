"""Cut a damage mask that follows the part on one clean photo.

SAM 2 turns a box into the part outline. The damage region is a new shape
inside that outline, so it is not one of the six training masks.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CHECKPOINT = "/workspace/cdf-anomalygen/checkpoints/facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def organic_blob(size, box, rng, coverage):
    width, height = size
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    span = math.sqrt(bw * bh * coverage)
    cx = int(rng.uniform(x0 + bw * 0.28, x0 + bw * 0.72))
    cy = int(rng.uniform(y0 + bh * 0.28, y0 + bh * 0.72))
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(7):
        angle = rng.random() * math.tau
        reach = span * rng.uniform(0.05, 0.35)
        ox = int(math.cos(angle) * reach)
        oy = int(math.sin(angle) * reach)
        rx = max(4, int(span * rng.uniform(0.28, 0.7)))
        ry = max(4, int(span * rng.uniform(0.22, 0.65)))
        draw.ellipse([cx + ox - rx, cy + oy - ry, cx + ox + rx, cy + oy + ry], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=1.4)).point(lambda p: 255 if p > 96 else 0)


def glass_shape(size, box, rng):
    x0, y0, x1, y1 = box
    if rng.random() < 0.45:
        inset_x = int((x1 - x0) * 0.04)
        inset_y = int((y1 - y0) * 0.04)
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [x0 + inset_x, y0 + inset_y, x1 - inset_x, y1 - inset_y], radius=8, fill=255)
        return mask
    return organic_blob(size, box, rng, rng.uniform(0.12, 0.32))


def dent_shape(size, box, rng):
    return organic_blob(size, box, rng, rng.uniform(0.06, 0.18))


def tire_shape(size, box, rng):
    x0, y0, x1, y1 = box
    bh = max(1, y1 - y0)
    cut = y0 + int(bh * rng.uniform(0.58, 0.82))
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x0, cut, x1, y1], fill=255)
    bulge = int(bh * rng.uniform(0.06, 0.14))
    draw.ellipse([x0, cut - bulge, x1, cut + bulge], fill=255)
    return mask


def clip_to_part(shape, part):
    shape_px = np.array(shape, dtype=np.uint8)
    part_px = np.array(part, dtype=np.uint8)
    kept = np.where((shape_px > 0) & (part_px > 0), 255, 0).astype(np.uint8)
    return Image.fromarray(kept, mode="L")


def part_from_box(image, box):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_sam2(CONFIG, CHECKPOINT, device="cuda"))
    array = np.array(image.convert("RGB"))
    x0, y0, x1, y1 = box
    with torch.inference_mode():
        predictor.set_image(array)
        masks, scores, _ = predictor.predict(
            box=np.array([x0, y0, x1, y1], dtype=np.float32),
            multimask_output=True,
        )
    order = np.argsort(scores)[::-1]
    for index in order:
        mask = masks[index]
        if int(mask.sum()) < 64:
            continue
        ys, xs = np.where(mask)
        center_y, center_x = (y0 + y1) / 2, (x0 + x1) / 2
        if ys.min() <= center_y <= ys.max() and xs.min() <= center_x <= xs.max():
            chosen = mask
            break
    else:
        chosen = masks[order[0]]
    return Image.fromarray(np.where(chosen, 255, 0).astype(np.uint8), mode="L")


def damage_mask(image, box, kind, seed):
    part = part_from_box(image, box)
    ys, xs = np.where(np.array(part) > 0)
    if len(xs) == 0:
        raise RuntimeError("SAM 2 returned an empty part.")
    part_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    rng = random.Random(seed)
    if kind == "shattered_glass":
        shape = glass_shape(image.size, part_box, rng)
    elif kind == "flat_tire":
        shape = tire_shape(image.size, part_box, rng)
    else:
        shape = dent_shape(image.size, part_box, rng)
    mask = clip_to_part(shape, part)
    if int(np.array(mask).sum() // 255) < 40:
        raise RuntimeError("The damage shape missed the part.")
    return part, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--box", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--kind", required=True, choices=["dent", "shattered_glass", "flat_tire", "bucket_dent"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--part", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    part, mask = damage_mask(image, tuple(args.box), args.kind, args.seed)
    args.part.parent.mkdir(parents=True, exist_ok=True)
    part.save(args.part)
    mask.save(args.mask)
    white = int(np.array(mask).sum() // 255)
    print(f"mask_pixels {white} size {image.size[0]}x{image.size[1]}", flush=True)


if __name__ == "__main__":
    main()
