"""Create a portable local comparison gallery from actual files only."""
import base64
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
plan = json.loads((root / "results/planning/reviewed-plan.json").read_text())
review_path = root / "results/visual-review.json"
review = json.loads(review_path.read_text()) if review_path.exists() else {}
mapping_path = root / "results/anomaly-output-map.json"
mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
status_path = root / "results/run-status.json"
status = json.loads(status_path.read_text()) if status_path.exists() else {}
sheet = Image.new("RGB", (960, len(plan["variants"]) * 230 + 45), "#eef1f5")
draw = ImageDraw.Draw(sheet)
headers = ["Clean base - Qwen via CDF", "Candidate damage edit - Qwen", "AnomalyGen output"]
for x, title in enumerate(headers):
    draw.text((x * 320 + 10, 12), title, fill="black")
rows = []
for i, variant in enumerate(plan["variants"]):
    name = variant["id"]
    paths = [root / f"results/clean/{name}.png", root / f"results/bootstrap/{name}.png",
        root / mapping[name] if name in mapping else root / f"results/not-generated/{name}.png"]
    cells = []
    for col, path in enumerate(paths):
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((312, 195))
            sheet.paste(image, (col * 320 + 4, i * 230 + 45))
            encoded = base64.b64encode(path.read_bytes()).decode()
            cells.append(f'<td><img src="data:image/png;base64,{encoded}" alt="{html.escape(name)}"></td>')
        else:
            missing = "Not generated" if col == 2 else "Not yet available locally"
            draw.text((col * 320 + 15, i * 230 + 110), missing, fill="#666666")
            cells.append(f"<td>{missing}</td>")
    label = f"{name} / requested: {variant['defect']}"
    draw.text((8, i * 230 + 244), label, fill="black")
    entry = review.get(name, {"status": "not reviewed"})
    assessment = html.escape(" · ".join(f"{key.replace('_', ' ')}: {value}" for key, value in entry.items()))
    rows.append(f'<tr><th colspan="3">{html.escape(label)}<p>{assessment}</p></th></tr><tr>{"".join(cells)}</tr>')
sheet.save(root / "results/comparison.png")
page = """<!doctype html><meta charset="utf-8"><title>Excavator Route 1 quality pilot</title>
<style>body{font:16px system-ui;margin:28px;background:#f5f6f8;color:#17212f}table{width:100%;border-collapse:collapse;table-layout:fixed}td{width:33.33%;padding:6px}img{width:100%;height:auto;cursor:zoom-in}th{text-align:left;padding:12px;background:#e5eaf0}p{font-weight:normal;font-size:14px;line-height:1.5}td{background:white}dialog{width:94vw;max-width:1600px;border:0;padding:12px}dialog::backdrop{background:#000b}dialog img{cursor:default}button{padding:8px 16px;margin-bottom:10px;cursor:pointer}</style>
<h1>Excavator Route 1 quality pilot</h1>
<p>One supplied reference photo. Clean bases and candidate damage edits use Qwen Image Edit through CDF Augmentation. Rejected candidates are excluded from AnomalyGen training; the original photo supplies real glass and dent crops. Only the right column contains actual AnomalyGen outputs. Missing outputs are shown explicitly. Masks and boxes are provisional; this is not an independently validated training dataset.</p>
<p>Click any image to inspect it at a larger size. The requested class describes the test case; the review states whether that damage is actually visible.</p>
<table><tr>""" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>" + "".join(rows) + """</table>
<dialog id="viewer"><button id="close">Close</button><img id="large" alt="Expanded comparison image"></dialog>
<script>
const viewer=document.getElementById('viewer');
document.querySelectorAll('td img').forEach(img=>img.addEventListener('click',()=>{document.getElementById('large').src=img.src;viewer.showModal();}));
document.getElementById('close').addEventListener('click',()=>viewer.close());
viewer.addEventListener('click',e=>{if(e.target===viewer)viewer.close();});
</script>"""
if status.get("summary"):
    page = page.replace("<table><tr>", "<p><strong>Run status:</strong> " + html.escape(status["summary"]) + "</p><table><tr>", 1)
(root / "results/gallery.html").write_text(page)
print("Wrote results/gallery.html and results/comparison.png")
