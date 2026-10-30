"""Local Qwen image-edit endpoint."""
import argparse
import base64
import io
import json
from pathlib import Path
import threading
import time

import torch
from diffusers import BitsAndBytesConfig, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from fastapi import FastAPI, Request
from PIL import Image
import uvicorn

ROOT = Path(__file__).resolve().parent
model = json.loads((ROOT / "results/edit-model.json").read_text())
settings_path = ROOT / "configs/edit-runtime.json"
settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
precision = settings.get("precision", "int8")
if precision not in {"int8", "nf4"}:
    raise ValueError(f"Unsupported transformer precision: {precision}")
quant = (BitsAndBytesConfig(load_in_8bit=True) if precision == "int8" else
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
torch.backends.cuda.matmul.allow_tf32 = True
transformer = QwenImageTransformer2DModel.from_pretrained(
    model["path"], subfolder="transformer", torch_dtype=torch.bfloat16,
    quantization_config=quant,
    device_map={"": "cuda"},
)
pipe = QwenImageEditPlusPipeline.from_pretrained(
    model["path"], transformer=transformer, torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.vae.enable_tiling()
lock = threading.Lock()
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ready", "model": model["model"], "quantization": precision}

@app.post("/v1/images/edits")
async def edit(request: Request):
    form = await request.form()
    upload = form.get("image") or form.get("image[]")
    if upload is None:
        from fastapi import HTTPException
        raise HTTPException(400, "multipart image is required")
    source = Image.open(io.BytesIO(await upload.read())).convert("RGB")
    prompt = str(form["prompt"])
    seed = int(form.get("seed", 42))
    steps = int(form.get("num_inference_steps", 30))
    # Read dimensions at request time so quality tuning does not reload 58 GB of weights.
    runtime = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    width = int(form.get("width", runtime.get("width", 1280)))
    height = int(form.get("height", runtime.get("height", 640)))
    true_cfg = float(form.get("true_cfg_scale", runtime.get("true_cfg_scale", 4.0)))
    started = time.time()
    with lock, torch.inference_mode():
        result = pipe(image=[source], prompt=prompt, negative_prompt=" ",
            width=width, height=height, num_inference_steps=steps,
            true_cfg_scale=true_cfg, guidance_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    receipt = {"model": model["model"], "revision": model["revision"],
        "seed": seed, "steps": steps, "quantization": precision,
        "width": width, "height": height, "true_cfg_scale": true_cfg,
        "elapsed_seconds": time.time() - started, "prompt": prompt}
    with (ROOT / "logs/edit-requests.jsonl").open("a") as f:
        f.write(json.dumps(receipt) + "\n")
    return {"created": int(time.time()), "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}],
        "metadata": receipt}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
