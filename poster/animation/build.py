"""Rebuild index.html from template.html + tracking.json (data inlined so the page works offline)."""
import json
from pathlib import Path
here = Path(__file__).parent
d = json.load(open(here / "tracking.json")); d["transformer"].pop("attention_matrix_layer1_storm", None)
(here / "index.html").write_text((here / "template.html").read_text().replace("/*DATA*/", json.dumps(d, separators=(",", ":"))))
print("built", here / "index.html")
