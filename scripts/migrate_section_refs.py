#!/usr/bin/env python3
"""Migrate document references in code comments to the current numbering of
docs/mathstrict_EN.md.

The code was documented against the archived layout (docs/docs_archieve/
mathstrict_EN copy.md). Every `§N` token and every `Keyword N` token
(Proposition, Lemma, Remark, Algorithm, Definition, Theorem, Section, ...)
is rewritten in a single left-to-right pass: whole-phrase rules and token
rules are alternatives of one regex, so a replacement is never re-scanned.
A token number is matched against the longest prefix present in the table
("10.4.7.5" -> "10.3.7.5" via the "10.4.7" key). Numbers that exist only
in the current layout, and tokens with no table entry, are left alone and
reported.

Usage:
    python3 scripts/migrate_section_refs.py            # rewrite in place
    python3 scripts/migrate_section_refs.py --check    # report only
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ["src", "tests"]
SUFFIXES = {".rs", ".wesl"}

# Old (archived) number -> current number. Prefixes match item numbers
# below them ("10.4.7" also covers "10.4.7.5"). Chapters 1, 3, 4, 11, 12
# kept their numbering and need no entries. Chapters 5-7 of the archive
# were removed; their few references are handled per file below.
MAP = {
    # Chapter 2 -> Section 10.0.x
    "2.8b": "10.0.2",
    "2.8c": "10.0.3",
    "2.8": "10.0",
    "2.9": "10.0.4",
    "2.10": "10.0.5",
    "2.11": "10.0.6",
    "2.12": "10.0.7",
    "2.13": "10.0.8",
    "2.14": "10.0.9",
    "2.15": "10.0.10",
    "2.16": "10.0.11",
    "2.17": "10.0.12",
    "2.17.1": "10.0.12.1",
    "2.17.2": "10.0.12.2",
    "2.17.3": "10.0.12.3",
    "2.17.4": "10.0.12.4",
    # Chapter 8 (radial Galerkin) -> the route (G) of Section 8.1 / A.3
    "8.0": "8.1",
    "8.0.6": "8.1.6",
    "8.2": "8.1.3",
    "8.5": "A.3",
    "8.7": "8.1.3",
    # Archived 9.10.x (density cores) -> current 9.2.x
    "9.10": "9.2",
    # Chapter 10 subsections shifted down by one
    "10.1": "10.1",
    "10.2": "10.1",
    "10.2.1": "10.1",
    "10.2.3": "10.1.1",
    "10.2.4": "10.1",
    "10.3": "10.2",
    "10.4": "10.3",
    "10.5": "10.4",
}

# Number overrides per file, on top of MAP, where the generic target would
# be wrong for that file.
FILE_MAP = {
    "src/scf/runout.rs": {"2.8": "10.0.2"},
    "src/scf/params.rs": {"2.8": "10.0.2"},
    "src/scf/transport.rs": {"2.8": "10.0.3", "9.2": "10.0.3"},
    "src/scf/quad.rs": {"9.4": "8.1.3"},
    "src/rtfp/shader/moments.wesl": {"8.3": "3", "9.4": "8.1.3"},
    "src/rheo/basis.rs": {"10.1": "8.1.3"},
}

# Whole-phrase rules (file -> {old: new}); "*" applies everywhere. These
# alternatives are tried before the token pattern at each position.
LITERAL = {
    "*": {"§2.8–2.16": "§10.0–10.0.11"},
    "src/scf/stokes/pd.rs": {"Proposition 10.4": "Section 10.3.2"},
    "src/rheo/basis.rs": {
        "Proposition 10.2": "Section 10.1",
        "mathstrict.md": "mathstrict_EN.md",
    },
    "src/rheo/reference.rs": {
        "Proposition 10.2": "Section 10.1",
        "Lemma 5.5": "Section 10.1",
    },
    "tests/support/rheo_selfcheck.rs": {"Proposition 10.2": "Section 10.1"},
}

KEYWORD = (
    r"§\s*|Propositions\s+|Proposition\s+|Lemma\s+|Remark\s+|Algorithm\s+"
    r"|Definition\s+|Theorem\s+|Corollary\s+|Section\s+|Hypothesis\s+"
)
TOKEN = re.compile(r"(" + KEYWORD + r")(\d+(?:\.\d+)*[a-z]?)")


def migrate(text: str, rel: str):
    """Return (new_text, replacements, unmapped) for one file."""
    lit = {**LITERAL["*"], **LITERAL.get(rel, {})}
    table = {**MAP, **FILE_MAP.get(rel, {})}
    alts = [re.escape(a) for a in sorted(lit, key=len, reverse=True)]
    pat = re.compile("|".join(alts + [TOKEN.pattern]) if alts else TOKEN.pattern)
    repl = []
    unmapped = []

    def lookup(num: str) -> str:
        n = num
        while n:
            if n in table:
                return table[n] + num[len(n):]
            n = n.rsplit(".", 1)[0] if "." in n else ""
        return ""

    def sub(m: re.Match) -> str:
        s = m.group(0)
        if s in lit:
            repl.append((s, lit[s]))
            return lit[s]
        tm = TOKEN.fullmatch(s)
        if not tm:
            return s
        new = lookup(tm.group(2))
        if not new:
            unmapped.append(s)
            return s
        out = tm.group(1) + new
        repl.append((s, out))
        return out

    return pat.sub(sub, text), repl, unmapped


def main() -> int:
    check = "--check" in sys.argv
    total = 0
    for d in SCAN_DIRS:
        for path in sorted((ROOT / d).rglob("*")):
            if path.suffix not in SUFFIXES:
                continue
            rel = path.relative_to(ROOT).as_posix()
            text = path.read_text()
            new, repl, unmapped = migrate(text, rel)
            if repl or unmapped:
                total += len(repl)
                print(f"{rel}: {len(repl)} replaced"
                      + (f", unmapped {sorted(set(unmapped))}" if unmapped else ""))
            if new != text and not check:
                path.write_text(new)
    print(f"{'would replace' if check else 'replaced'} {total} references")
    return 0


if __name__ == "__main__":
    sys.exit(main())
