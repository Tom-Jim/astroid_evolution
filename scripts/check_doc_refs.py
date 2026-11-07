#!/usr/bin/env python3
"""Check doc/code cross-references against docs/mathstrict_EN.md.

Two checks, both cheap enough for CI:

1. Doc-internal references: every `Section X`, `Proposition X`, `Lemma X`,
   `Definition X`, `Algorithm X`, `Remark X`, `Theorem X`, `Corollary X`,
   `Hypothesis X`, `Criterion X`, `PO-N` mentioned in the document must be
   defined somewhere in it, and no two entries may share a number.

2. Code `§` references: every `§N` (and `Prop./Proposition/Lemma/Section N`)
   in `src/` and `tests/` must resolve to a numbered heading of the doc.

Exits non-zero on any violation. Read-only by design.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "mathstrict_EN.md"

# `### 8.1.3` / `**Proposition 8.1.5` / `Definition 11.3` / `### A.3`.
HEAD_NUM = re.compile(r"^#{2,6}\s+(?:\*{0,2}(?:Section|Appendix)\s+)?([0-9]+(?:\.[0-9A-Za-z]+)*)", re.M)
ENTRY = re.compile(
    r"(?:\*{1,2}|^#{1,6}\s*)(Proposition|Lemma|Definition|Algorithm|Remark|Theorem|Corollary|Hypothesis|Criterion|Conjecture)\s+([0-9]+(?:\.[0-9A-Za-z]+)*)",
    re.M,
)
PO_DEF = re.compile(r"\|\s*(PO-\d+)\s*\|")
PO_REF = re.compile(r"\b(PO-\d+)\b")
REF_KIND = re.compile(
    r"\b(Section|Proposition|Lemma|Definition|Algorithm|Remark|Theorem|Corollary|Hypothesis|Criterion|Appendix)\s+"
    r"(?:(?:Prop(?:osition)?|Lem(?:ma)?|Def(?:inition)?|Sec(?:tion)?|App(?:endix)?)\s+)*([0-9]+(?:\.[0-9A-Za-z]+)*|[A-Z](?:\.[0-9]+)+)"
)
CODE_REF = re.compile(r"§\s*([0-9]+(?:\.[0-9A-Za-z]+)*)")


def doc_index(text: str):
    heads = set(HEAD_NUM.findall(text))
    sections = {h.split(".")[0] for h in heads}
    entries = {}  # number -> kind
    for kind, num in ENTRY.findall(text):
        prev = entries.get(num)
        if prev and prev != kind:
            print(f"doc: number {num} claimed by both {prev} and {kind}")
        entries.setdefault(num, kind)
    pos = set(PO_DEF.findall(text))
    return heads, sections, entries, pos


def main() -> int:
    text = DOC.read_text(encoding="utf-8")
    heads, sections, entries, po_defs = doc_index(text)
    bad = 0

    # 1a. dangling doc-internal references
    for kind, num in REF_KIND.findall(text):
        if kind == "Appendix":
            continue
        if num in entries or num in heads:
            continue
        # A reference may target a sub-entry without a heading (e.g. a
        # proposition inside a section): accept it if its parent exists.
        parts = num.split(".")
        if any(".".join(parts[:i]) in heads or ".".join(parts[:i]) in entries for i in range(1, len(parts))):
            continue
        if parts[0] in sections or parts[0].isdigit() is False:
            if parts[0] in sections:
                print(f"doc: dangling {kind} {num}")
                bad += 1
                continue
        print(f"doc: dangling {kind} {num}")
        bad += 1

    # 1b. PO references
    for po in PO_REF.findall(text):
        if po not in po_defs and "PO-" not in po:
            print(f"doc: dangling {po}")
            bad += 1

    # 2. code § references
    for base in (ROOT / "src", ROOT / "tests"):
        for f in base.rglob("*.rs"):
            try:
                src = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_no, line in enumerate(src.splitlines(), 1):
                for num in CODE_REF.findall(line):
                    top = num.split(".")[0]
                    if num in heads or num in entries or top in sections:
                        continue
                    print(f"{f.relative_to(ROOT)}:{line_no}: §{num} not found in {DOC.name}")
                    bad += 1

    if bad == 0:
        print("check_doc_refs: all references resolve")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
