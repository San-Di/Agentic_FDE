"""Record an image-grounded dataset plan using the existing Cosmos Reason2 service."""
import base64
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
prompt = """You are planning a small synthetic rental-equipment damage dataset from ONE reference photo.
Inspect the photo carefully. User wants different excavator types, backgrounds and viewpoints,
with exactly one visible damage category per image: flat_tire, shattered_glass, or dent.
Flat tires require a WHEELED excavator, never caterpillar tracks. All other parts must be intact.
We will use image editing for clean base variations, then train Cosmos AnomalyGen with annotated
defect examples. No Omniverse. The supplied photo is a damaged reference, NOT a clean base.
Verified source facts: this excavator has caterpillar tracks and NO rubber tires. Glass and body
are both damaged. Do not invent a flat tire in the source.
Return compact JSON only, under 1000 words, with: source_observations; observed_defects; absent_examples; limitations;
clean_repair_prompt; six variants (id, excavator_type, background, viewpoint, target_defect,
clean_generation_prompt, defect_edit_prompt, target_part); quality_checks.
Include two variants per class and realistic parked rental machinery. Remove the person, text
and logo in newly synthesized bases. Do not pretend one image supplies real examples of all classes.
Synthetic bootstrapped training examples must be identified as synthetic, and all labels provisional.
Each prompt must be under 35 words. No lists of mechanical systems. Never repeat sentences.
Prompts must describe physically visible damage, not an accident narrative.
"""
image = base64.b64encode((ROOT / "inputs/reference.jpg").read_bytes()).decode()
request = {
    "model": "cosmos-reason2", "temperature": 0.6, "max_tokens": 2400,
    "repetition_penalty": 1.15, "top_p": 0.9,
    "response_format": {"type": "json_object"},
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}},
    ]}],
}
output = ROOT / "results/planning"
output.mkdir(parents=True, exist_ok=True)
(output / "prompt.txt").write_text(prompt)
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=360) as response:
    result = json.load(response)
(output / "response.json").write_text(json.dumps(result, indent=2))
content = result["choices"][0]["message"]["content"]
(output / "response.txt").write_text(content)
start = content.find("{", content.find("</think>") + len("</think>")) if "</think>" in content else content.find("{")
plan, _ = json.JSONDecoder().raw_decode(content[start:])
(output / "plan.json").write_text(json.dumps(plan, indent=2))
print(json.dumps(plan, indent=2))
