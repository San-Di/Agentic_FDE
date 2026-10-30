"""Turn studio settings into one cell per machine, damage, view, and background."""
from __future__ import annotations

MACHINES = [
    {
        "id": "tracked",
        "label": "Tracked excavator",
        "texture": "TrackedExcavator",
        "ready": True,
        "donor": "results/clean/dent_depot.png",
        "applies": ["dent", "shattered_glass", "bucket_dent"],
        "detail": "Steel tracks. Body-dent and shattered-window adapters are already trained.",
    },
    {
        "id": "wheeled",
        "label": "Wheeled excavator",
        "texture": "WheeledExcavator",
        "ready": True,
        "donor": "results/clean/tire_yard.png",
        "applies": ["flat_tire"],
        "detail": "Four rubber tires. The flat-tire adapter is already trained.",
    },
    {
        "id": "dump",
        "label": "Articulated dump truck",
        "texture": "DumpTruck",
        "ready": False,
        "donor": None,
        "applies": ["dent", "flat_tire", "shattered_glass", "bucket_dent"],
        "detail": "Kept in the request. Generation waits until a truck adapter is trained.",
    },
    {
        "id": "haul",
        "label": "Rigid haul truck",
        "texture": "HaulTruck",
        "ready": False,
        "donor": None,
        "applies": ["dent", "flat_tire", "shattered_glass", "bucket_dent"],
        "detail": "Kept in the request. Generation waits until a truck adapter is trained.",
    },
]

DAMAGES = [
    {
        "id": "dent",
        "label": "Dents",
        "defect": "dent",
        "ready": True,
        "anomaly_type": "TrackedExcavator+dent",
        "roi_prompt": "excavator side body panel",
        "detail": "Trained on tracked excavators.",
    },
    {
        "id": "flat_tire",
        "label": "Flat tires",
        "defect": "flat_tire",
        "ready": True,
        "anomaly_type": "WheeledExcavator+flat_tire",
        "roi_prompt": "excavator rubber tire",
        "detail": "Trained on wheeled excavators.",
    },
    {
        "id": "shattered_glass",
        "label": "Shattered windows",
        "defect": "shattered_glass",
        "ready": True,
        "anomaly_type": "TrackedExcavator+shattered_glass",
        "roi_prompt": "excavator cab window",
        "detail": "Trained on tracked excavators.",
    },
    {
        "id": "bucket_dent",
        "label": "Dent on the claws / buckets",
        "defect": "bucket_dent",
        "ready": False,
        "anomaly_type": None,
        "roi_prompt": "the digging bucket and its teeth, excluding the boom and the ground",
        "detail": "Kept in the request. Generation waits until this adapter is trained.",
    },
]

SUNBURST_APPLIES = {
    "tracked": ["dent", "shattered_glass", "bucket_dent"],
    "wheeled": ["dent", "flat_tire", "shattered_glass", "bucket_dent"],
    "dump": ["dent", "flat_tire", "shattered_glass", "bucket_dent"],
    "haul": ["dent", "flat_tire", "shattered_glass", "bucket_dent"],
}

LIMITS = {"varieties": 4, "positions": 6, "backgrounds": 6, "images": 48}

DEFAULTS = {
    "varieties": 2,
    "machines": ["tracked", "wheeled"],
    "damages": ["dent", "flat_tire", "shattered_glass"],
    "positions": ["side view", "three-quarter view", "front view"],
    "backgrounds": [
        "gravel rental yard",
        "earthmoving construction site",
        "quarry parking area",
    ],
}

SAMPLE_PROMPT = (
    "Rental excavators and big trucks. Build a labeled damage dataset for return "
    "inspection from this photo: dents, flat tires, shattered windows, and a dent "
    "on the bucket, across different machines, viewpoints, and places."
)


def catalog():
    return {
        "machines": MACHINES,
        "damages": DAMAGES,
        "defaults": DEFAULTS,
        "limits": LIMITS,
        "sample_prompt": SAMPLE_PROMPT,
        "sample_photo": "inputs/reference.jpg",
    }


def _lines(value, fallback, limit):
    if isinstance(value, str):
        items = [line.strip() for line in value.splitlines()]
    elif isinstance(value, list):
        items = [str(line).strip() for line in value]
    else:
        items = list(fallback)
    items = [item for item in items if item][:limit]
    return items or list(fallback)


def _hold_reason(machine, damage):
    if not machine["ready"] and not damage["ready"]:
        return f"{machine['label']} and {damage['label'].lower()} both wait for a trained adapter."
    if not machine["ready"]:
        return f"{machine['label']} waits for a trained adapter."
    return f"{damage['label']} waits for a trained adapter."


def expand(settings):
    """Return the cells this run will generate and the cells it will only record."""
    settings = settings or {}
    pipeline = settings.get("pipeline") if settings.get("pipeline") in {"anomalygen", "sunburst", "gptclean"} else "anomalygen"
    mask_pipeline = pipeline in {"sunburst", "gptclean"}
    chosen_machines = set(settings.get("machines") or [])
    chosen_damages = set(settings.get("damages") or [])
    machines = [item for item in MACHINES if item["id"] in chosen_machines]
    try:
        varieties = int(settings.get("varieties") or len(machines) or 1)
    except (TypeError, ValueError):
        varieties = 1
    varieties = max(1, min(LIMITS["varieties"], varieties))
    machines = machines[:varieties]
    damages = [item for item in DAMAGES if item["id"] in chosen_damages]
    positions = _lines(settings.get("positions"), DEFAULTS["positions"], LIMITS["positions"])
    backgrounds = _lines(settings.get("backgrounds"), DEFAULTS["backgrounds"], LIMITS["backgrounds"])
    try:
        cap = int(settings.get("total") or 0)
    except (TypeError, ValueError):
        cap = 0
    cap = max(0, min(LIMITS["images"], cap))

    generate, held = [], []
    for machine in machines:
        applies = SUNBURST_APPLIES[machine["id"]] if mask_pipeline else machine["applies"]
        applicable = [damage for damage in damages if damage["id"] in applies]
        for damage in applicable:
            ready = True if mask_pipeline else bool(machine["ready"] and damage["ready"])
            for position in positions:
                for background in backgrounds:
                    cell = {
                        "machine": machine["id"],
                        "machine_label": machine["label"],
                        "texture": machine["texture"],
                        "donor": machine["donor"],
                        "damage": damage["id"],
                        "damage_label": damage["label"],
                        "defect": damage["defect"],
                        "anomaly_type": damage["anomaly_type"],
                        "roi_prompt": damage["roi_prompt"],
                        "position": position,
                        "background": background,
                        "ready": ready,
                        "reason": None if ready else _hold_reason(machine, damage),
                    }
                    (generate if ready else held).append(cell)
    full_generate = list(generate)
    full = len(full_generate)
    if cap and cap < full:
        generate = generate[:cap]
    scenes = []
    seen = set()
    for cell in generate:
        key = (cell["machine"], cell["position"], cell["background"])
        if key in seen:
            continue
        seen.add(key)
        scenes.append({
            "machine": cell["machine"],
            "machine_label": cell["machine_label"],
            "texture": cell["texture"],
            "donor": cell["donor"],
            "position": cell["position"],
            "background": cell["background"],
        })
    by_machine = {}
    for cell in full_generate:
        names = by_machine.setdefault(cell["machine_label"], [])
        if cell["damage_label"] not in names:
            names.append(cell["damage_label"])
    summary = []
    for label, names in by_machine.items():
        position_word = "position" if len(positions) == 1 else "positions"
        background_word = "background" if len(backgrounds) == 1 else "backgrounds"
        summary.append(
            f"{label}: {', '.join(names)}. {len(positions)} {position_word} × {len(backgrounds)} {background_word}."
        )
    if cap and cap < full:
        summary.append(f"This run keeps the first {len(generate)} of {full}. Raise Total images to keep the rest.")
    if held and not mask_pipeline:
        summary.append(f"Held until a fine-tune exists: {len(held)} combination{'s' if len(held) != 1 else ''}.")
    if not machines:
        summary.append("Choose at least one machine type.")
    elif not damages:
        summary.append("Choose at least one damage type.")
    elif not generate and not held:
        summary.append("None of the checked damages apply to the checked machines.")
    if pipeline == "gptclean":
        minutes = round(len(scenes) * 1 + len(generate) * 1) if generate else 1
    else:
        per_image = 2.5 if pipeline == "sunburst" else 1.2
        minutes = round(len(scenes) * 4 + len(generate) * per_image) if generate else 1
    return {
        "pipeline": pipeline,
        "generate": generate,
        "held": held,
        "scenes": scenes,
        "summary": summary,
        "full_total": full,
        "total": len(generate),
        "held_total": len(held),
        "scene_total": len(scenes),
        "minutes": minutes,
        "positions": positions,
        "backgrounds": backgrounds,
        "varieties": len(machines),
        "cap": cap or full,
    }
