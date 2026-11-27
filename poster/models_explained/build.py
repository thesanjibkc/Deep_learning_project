"""Build index.html from template.html + tutorial_data.json (python3 scripts/export_tutorial.py first)."""
from pathlib import Path
h = Path(__file__).parent
(h / "index.html").write_text((h / "template.html").read_text().replace("__DATA__", (h / "tutorial_data.json").read_text()))
print("built", h / "index.html")
