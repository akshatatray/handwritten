"""Apply label corrections to a sample's labels.txt: set_labels.py SAMPLE_DIR wid=split [wid=split ...]

Lines for words not mentioned keep their pre-filled guess. A word that was labelled automatically
can be corrected or skipped (wid=?) the same way. Used by Claude after reading the sheets."""
import json
import sys
from pathlib import Path

sdir = Path(sys.argv[1]).expanduser()
fixes = dict(arg.split("=", 1) for arg in sys.argv[2:])
path = sdir / "labels.txt"
out, seen = [], set()
for raw in path.read_text(encoding="utf-8").splitlines():
    parts = raw.split()
    if len(parts) >= 3 and not raw.startswith("#") and parts[0] in fixes:
        out.append(f"{parts[0]} {parts[1]} {fixes[parts[0]]}")
        seen.add(parts[0])
    else:
        out.append(raw)
# Words labelled automatically aren't listed; add them so the correction overrides the automatic label.
seg = json.loads((sdir / "segments.json").read_text(encoding="utf-8"))
for wid in sorted(set(fixes) - seen):
    if wid not in seg["words"]:
        sys.exit(f"ERROR: no word {wid} in this sample")
    out.append(f"{wid} {len(seg['words'][wid]['pieces'])} {fixes[wid]}")
    seen.add(wid)
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"{len(seen)} labels updated in {path}")
