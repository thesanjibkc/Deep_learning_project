"""Rebuild index.html from template.html + zoo_data.json (data inlined, so the page works offline)."""
import json
from pathlib import Path
here = Path(__file__).parent
data = json.load(open(here / "zoo_data.json"))
(here / "index.html").write_text((here / "template.html").read_text().replace("/*DATA*/", json.dumps(data, separators=(",", ":"))))
print("built", here / "index.html")
