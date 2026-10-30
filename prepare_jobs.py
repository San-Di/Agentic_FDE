"""Compile the pilot plan into CDF Augmentation configs."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
(ROOT / "configs").mkdir(exist_ok=True)
(ROOT / "results/clean").mkdir(parents=True, exist_ok=True)
(ROOT / "results/bootstrap").mkdir(parents=True, exist_ok=True)
gateway = subprocess.check_output(["docker", "network", "inspect", "cdf-route1",
    "--format", "{{(index .IPAM.Config 0).Gateway}}"], text=True).strip()
variants = [
    ("tire_yard", "flat_tire", "a compact yellow wheeled excavator with four rubber tires and no tracks", "a gravel rental yard", "front left three-quarter view at wheel height"),
    ("tire_road", "flat_tire", "a large orange wheeled excavator with four rubber tires and no tracks", "a closed paved roadside work area", "left side view at wheel height"),
    ("glass_site", "shattered_glass", "a medium yellow tracked excavator", "an earthmoving construction site", "front left three-quarter view close enough to inspect the cab glass"),
    ("glass_farm", "shattered_glass", "a compact white tracked mini excavator with an enclosed glass cab", "a farm equipment parking area", "left side view close enough to inspect the cab window"),
    ("dent_depot", "dent", "a large orange tracked excavator", "a concrete equipment depot", "rear left three-quarter view showing the side engine housing and cab"),
    ("dent_quarry", "dent", "a medium blue tracked excavator", "a gravel quarry parking area", "left side view showing the engine cover and cab"),
]
defect_prompts = {
    "flat_tire": "Deflate ONLY the nearest visible rubber tire. Its lower sidewall must visibly collapse and bulge flat against the ground, with the rim lowered. Keep the rim undamaged. All other tires remain fully inflated; glass and metal panels remain undamaged.",
    "shattered_glass": "Shatter ONLY the large nearest cab window: dense realistic spiderweb cracks with a small missing-glass opening and jagged glass edges still held by the black window frame. Other windows remain clear and intact; the metal body and running gear remain undamaged.",
    "dent": "Add ONLY one deep concave impact dent with sharp creases in the large painted metal side engine cover behind the cab. The panel remains attached. Use physically shaded folded metal, not painted markings. All glass and running gear remain intact.",
}
common = "Natural rental inspection photograph, realistic machinery construction and photographic lighting. No people, logos, writing, arrows or labels."
repair_prompt = (
    "Repair the excavator in this photograph completely. Replace every shattered cab window with intact clear glass. "
    "Rebuild the crushed yellow engine housing and rear panels as smooth intact painted metal enclosing the radiator. "
    "Remove the worker, clipboard and top-right logo. Preserve the tracked excavator and camera view. " + common
)
jobs = [{"id": "clean_repaired", "stage": "clean", "source": str(ROOT / "inputs/reference.jpg"),
         "output": str(ROOT / "results/clean/repaired.png"), "prompt": repair_prompt, "seed": 100}]
reviewed = []
for i, (name, defect, machine, background, view) in enumerate(variants):
    prompt = (f"Use the reference as a machinery photography guide. Create a new photograph of {machine}, parked in {background}, "
        f"seen from a {view}. Show the entire machine's cab, chassis and running gear. "
        "All glass must be clear and unbroken, all panels undented and closed, all tires fully inflated. " + common)
    clean = ROOT / f"results/clean/{name}.png"
    jobs.append({"id": name, "stage": "clean", "source": str(ROOT / "results/clean/repaired.png"),
        "output": str(clean), "prompt": prompt, "seed": 200 + i})
    jobs.append({"id": name + "_bootstrap", "stage": "bootstrap", "source": str(clean),
        "output": str(ROOT / f"results/bootstrap/{name}.png"),
        "prompt": "Keep this exact photograph, viewpoint, equipment and background. " + defect_prompts[defect] + " " + common,
        "seed": 300 + i})
    reviewed.append({"id": name, "defect": defect,
        "texture": "WheeledExcavator" if defect == "flat_tire" else "TrackedExcavator",
        "machine": machine, "background": background,
        "view": view, "training_reference_provenance": "Qwen synthetic bootstrap; not additional real evidence"})
for job in jobs:
    out = Path(job["output"])
    config = {
        "data": [{"inputs": {"rgb": job["source"]}, "output": {
            "video": str(out), "caption": str(out.with_suffix(".txt")), "metadata": str(out.with_suffix(".json"))}}],
        "endpoints": [{"id": "image-edit", "role": "image_edit", "url": f"http://{gateway}:8011/v1",
            "model": "Qwen/Qwen-Image-Edit-2511", "adapter": "openai.images.edits", "timeout": 1200}],
        "captioning": {"llm": {"text": job["prompt"]}},
        "augmentation": {"model": {"name": "image-edit"}, "parameters": {"seed": job["seed"], "num_inference_steps": 30}},
        "pipeline": {"retry": 0},
    }
    # JSON is a YAML subset and avoids adding dependencies to the driver.
    path = ROOT / "configs" / (job["id"] + ".yaml")
    path.write_text(json.dumps(config, indent=2))
    job["config"] = str(path)
(ROOT / "jobs.json").write_text(json.dumps(jobs, indent=2))
(ROOT / "results/planning/reviewed-plan.json").write_text(json.dumps({
    "source": "One supplied real photograph; two visible defects on a tracked machine",
    "model_plan_review": "Planner produced only two variants; agent expanded to required six and removed extra rim deformation from flat-tire prompt.",
    "variants": reviewed, "labels": "Provisional masks and boxes; validate visible output before training a detector",
}, indent=2))
print(f"Prepared {len(jobs)} CDF image-edit jobs; endpoint gateway {gateway}")
