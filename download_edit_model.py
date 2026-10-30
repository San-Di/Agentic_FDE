import json
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
revision = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
path = snapshot_download("Qwen/Qwen-Image-Edit-2511", revision=revision,
    token=(Path.home() / ".hf_token").read_text().strip(), max_workers=4)
(root / "results/edit-model.json").write_text(json.dumps({
    "model": "Qwen/Qwen-Image-Edit-2511", "revision": revision, "path": path,
}, indent=2))
print(path)
