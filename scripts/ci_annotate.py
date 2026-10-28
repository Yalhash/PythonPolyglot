"""Turn tool output into GitHub annotations, grouped per file.

GitHub shows at most 10 error annotations per step, so messages are grouped by file:
one annotation carries every problem in that file.

    mypy | python scripts/ci_annotate.py lines          # "path:line: error: msg"
    ruff check --output-format=concise | ... lines      # "path:line:col: CODE msg"
    python scripts/ci_annotate.py junit junit.xml       # pytest failures
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

LINE = re.compile(r"^(?P<path>[^:\s][^:]*\.py):(?P<line>\d+):(?:\d+:)?\s*(?P<msg>.*)$")


def esc(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def emit(groups: dict[str, list[tuple[int, str]]], title: str) -> int:
    for path, items in sorted(groups.items()):
        first = min(line for line, _ in items)
        body = "\n".join(f"L{line}: {msg}" for line, msg in sorted(items))[:20000]
        print(f"::error file={path},line={first},title={title} ({len(items)})::{esc(body)}")
    return sum(len(v) for v in groups.values())


def lines() -> int:
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for raw in sys.stdin:
        sys.stdout.write(raw)
        m = LINE.match(raw.strip())
        if m and "note:" not in m["msg"][:6]:
            groups[m["path"].replace("\\", "/")].append((int(m["line"]), m["msg"]))
    return emit(groups, sys.argv[2] if len(sys.argv) > 2 else "problems")


def junit(path: str) -> int:
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    root = ET.parse(path).getroot()
    for case in root.iter("testcase"):
        for tag in ("failure", "error"):
            node = case.find(tag)
            if node is None:
                continue
            file = case.get("file") or case.get("classname", "").replace(".", "/") + ".py"
            text = (node.get("message") or "") + "\n" + (node.text or "")
            groups[file].append((int(case.get("line") or 1), f"{case.get('name')}: {text[-3000:]}"))
    return emit(groups, "test failures")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "lines":
        lines()
    elif mode == "junit":
        junit(sys.argv[2])
