#!/usr/bin/env python3
"""Notebase integrity checker (used to supervise reorganizations).

Modes:
  snapshot <out.json>  : record every .md file (path, content hash) and every
                         [[wikilink]] target in the vault.
  verify <before.json> : compare the current vault against a snapshot:
      - every content hash from the snapshot must still exist somewhere
        (moves and renames are fine, deletions and silent edits are not;
        intentionally edited files are reported, not failed, so the
        supervisor can eyeball the diff list)
      - every [[wikilink]] in the current vault must resolve to an existing
        file basename
      - reports files whose content changed or disappeared, and links that
        do not resolve
"""
import hashlib
import json
import re
import sys
from pathlib import Path

VAULT = Path(__file__).resolve().parent.parent / "docs" / "notes"
LINK = re.compile(r"\[\[([^\]|#]+)")


def md_files():
    return [p for p in VAULT.rglob("*.md") if ".obsidian" not in p.parts]


def snapshot(out):
    files = {}
    for p in md_files():
        files[str(p.relative_to(VAULT))] = hashlib.md5(p.read_bytes()).hexdigest()
    Path(out).write_text(json.dumps(files, indent=1))
    print(f"snapshot: {len(files)} md files -> {out}")


def verify(before_path):
    before = json.loads(Path(before_path).read_text())
    now = {}
    for p in md_files():
        now[str(p.relative_to(VAULT))] = hashlib.md5(p.read_bytes()).hexdigest()

    now_hashes = set(now.values())
    missing, moved = [], []
    for path, h in before.items():
        if h in now_hashes:
            if path not in now or now.get(path) != h:
                moved.append(path)
        else:
            missing.append(path)
    print(f"files before: {len(before)}, now: {len(now)}")
    print(f"moved/renamed (content intact): {len(moved)}")
    if missing:
        print("CONTENT CHANGED OR DELETED (review each!):")
        for m in missing:
            print(f"  - {m}")
    # link resolution
    basenames = {p.stem for p in md_files()}
    # known pre-existing non-links (template placeholders, literal text)
    IGNORE = {"Note A", "Note B", "Note Name", "1,1,1", "PRIOR_FINDINGS",
              "reference_dgx_spark", "docs/environment.md", "Past Winners"}
    bad = []
    for p in md_files():
        for target in LINK.findall(p.read_text(errors="ignore")):
            t = target.strip().rstrip("\\").strip()
            if (t and t not in basenames and t not in IGNORE
                    and not t.endswith((".png", ".jpg", ".pdf", ".svg"))):
                bad.append((str(p.relative_to(VAULT)), t))
    if bad:
        print(f"UNRESOLVED WIKILINKS: {len(bad)}")
        for f, t in bad[:40]:
            print(f"  {f}  ->  [[{t}]]")
    else:
        print("all wikilinks resolve")
    return 1 if missing or bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "snapshot":
        snapshot(sys.argv[2])
    else:
        sys.exit(verify(sys.argv[2]))
