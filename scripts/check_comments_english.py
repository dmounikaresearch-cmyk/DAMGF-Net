"""Report comments and strings that contain non-ASCII (possibly non-English) text.

Usage:
    python scripts/check_comments_english.py src scripts

PAA requires all code comments to be in English. This script lists every
line in .py and .ipynb files that contains non-ASCII characters so they can
be reviewed and translated before the repository is made public.
"""

import json
import pathlib
import sys


def scan_lines(path, lines):
    for no, line in enumerate(lines, 1):
        if any(ord(ch) > 127 for ch in line):
            print(f"{path}:{no}: {line.rstrip()}")


def main(roots):
    for root in roots or ["."]:
        for path in pathlib.Path(root).rglob("*"):
            if path.suffix == ".py":
                scan_lines(path, path.read_text(encoding="utf-8", errors="replace").splitlines())
            elif path.suffix == ".ipynb":
                nb = json.loads(path.read_text(encoding="utf-8"))
                for i, cell in enumerate(nb.get("cells", [])):
                    scan_lines(f"{path}[cell {i}]", "".join(cell.get("source", [])).splitlines())


if __name__ == "__main__":
    main(sys.argv[1:])
