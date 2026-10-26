"""The docs describe the code they cite, link what exists, and every complete document they show compiles.

Offline, on CPU, no model loaded, in seconds: `uv run pytest tests/test_docs.py -q`. It reads every page under
`docs/` and the root guides (`CLAUDE.md`, `AGENTS.md`, `README.md`, `FORMAT.md`) and holds them to these rules:

* every relative link resolves inside the repo, and every `#anchor` is a heading of its target by GitHub's slug
  (`slug`; the FORMAT.md heading ### `data.<name>` is #dataname);
* every page under `docs/` opens with YAML front matter holding `title`, `one_liner` and `tags`, and the files its
  `related:` and `sources:` lists name exist;
* a repo path in backticks exists: one under `clab/`, `tests/`, `documents/`, `data/` or `docs/` with a file
  extension or a trailing slash, any other `.py` path with a slash (read from the repo root or from `clab/`), and a
  root file named in capitals (`FORMAT.md`) or `pyproject.toml`;
* a symbol cited beside its `.py` file (`clab/compile/positions.py` `landing`, and the other forms
  `PATH_THEN_SYMBOL` and `SYMBOL_THEN_PATH` accept) is defined in that file: a def, a class, an assignment or an
  import;
* nothing is cited by line number (`plan.py:123`, `#L123`): line numbers drift, so code is cited by path and
  symbol;
* none of `BANNED` appears, case-insensitively: machine and user names, home paths, and planning files that are
  temporary or gone;
* every fenced ```json block that holds an object with both `model` and `steps` is a complete document, and
  compiles with the document alone: it is written into the suite's copy of the corpus (`corpus` in conftest.py,
  whose `data/` is the repo's and whose `documents/artifacts/` holds the stand-in bundles the corpus documents
  load), and `Plan(path).compile()` must succeed. A ```json block with both keys that is not valid JSON fails too:
  complete it, or drop `model` to make it a fragment. Fragments are not compiled.

Fenced code is not prose: links, paths, symbols, line numbers and banned terms are checked outside it only.
"""

from __future__ import annotations

import functools
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import ROOT as ROOT_DIR

from clab.plan import Plan

ROOT = Path(ROOT_DIR)
GUIDES = ("CLAUDE.md", "AGENTS.md", "README.md", "FORMAT.md")

# Citations by line number allowed to remain. It is 0, and never rises.
LINE_CITATION_BASELINE = 0
LINE_CITATION = re.compile(r"\.(?:py|sh|yml|yaml|toml|json|md)(?::\d|#L\d)")

# Terms no page may carry, as case-insensitive patterns. One is written in two parts so that a search of the repo
# for it finds nothing, this file included.
BANNED = (
    r"hakone", r"bippu", r"localjadenfk", r"\bndif2\b", r"/disk/",
    r"\b(?:mini|ana)?conda",            # conda, miniconda, anaconda; not "secondary"
    r"~/wd", "causa" "lab", r"\bCHARTER\.md", r"\bROADMAP\.md", r"\bNNTERP\.md", r"\bPLAN\.md", r"\bDESIGN\.md",
    r"clab-trials",
)

EXTENSIONS = ("py", "json", "md", "toml", "txt", "yml", "yaml", "cfg", "ini", "sh")
REPO_DIRS = ("clab/", "tests/", "documents/", "data/", "docs/")
# Paths that name files in other packages, not this repo.
EXTERNAL = ("nnsight/", "nnterp/", "transformers/", "torch/", "src/")
FILE_TOKEN = re.compile(r"^[\w.\-/]*[\w\-]\.(?:" + "|".join(EXTENSIONS) + r")$")
DIR_TOKEN = re.compile(r"^(?:" + "|".join(REPO_DIRS) + r")(?:[\w.\-]+/)*$")
ROOT_FILE = re.compile(r"^(?:[A-Z][A-Z0-9_\-]*\.md|pyproject\.toml)$")

SYMBOL = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\(\))?"
PY_PATH = r"[\w.\-/]*\.py(?::\d+(?:-\d+)?)?"
# `path.py` `Symbol`, `path.py`'s `Symbol`, `path.py`: `Symbol`, (`path.py`, `Symbol`)
PATH_THEN_SYMBOL = re.compile(rf"(?=`({PY_PATH})`(?:'s)?[ \t]*[,:]?[ \t]*`({SYMBOL})`)")
# `Symbol` (`path.py`), `Symbol`, `path.py`, `Symbol` in `path.py`, `Symbol` at `path.py`, `Symbol` — `path.py`.
# Both look ahead, so the matches overlap: in `a.py`, `b.py` `Name` the second pair is found too.
SYMBOL_THEN_PATH = re.compile(rf"(?=`({SYMBOL})`[ \t]*(?:\(|,|in|at|—|-)[ \t]*`({PY_PATH})`)")

CODE_SPAN = re.compile(r"(`+)(.+?)\1")
LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_ANCHOR = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")
HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
FENCE_OPEN = re.compile(r"^\s*(`{3,}|~{3,})\s*([^\s`]*)")
COMPLETE = re.compile(r"\"model\"\s*:[\s\S]*\"steps\"\s*:|\"steps\"\s*:[\s\S]*\"model\"\s*:")


def doc_files() -> list[Path]:
    return sorted((ROOT / "docs").rglob("*.md")) + [ROOT / name for name in GUIDES if (ROOT / name).exists()]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


@functools.lru_cache(maxsize=None)
def parsed(path: Path) -> tuple[tuple[tuple[int, str], ...], tuple[tuple[int, str, str], ...]]:
    """A page as (its prose lines, its fenced blocks): each prose line with its number, each fenced block as the
    number of its opening line, the first word of its info string, and its text."""
    prose: list[tuple[int, str]] = []
    fences: list[tuple[int, str, str]] = []
    fence: tuple[str, int, str] | None = None          # the opening marker, its line, its info word
    body: list[str] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if fence is None:
            opening = FENCE_OPEN.match(line)
            if opening:
                fence, body = (opening.group(1), number, opening.group(2)), []
            else:
                prose.append((number, line))
        elif re.match(rf"^\s*{re.escape(fence[0][0])}{{{len(fence[0])},}}\s*$", line):
            fences.append((fence[1], fence[2], "\n".join(body)))
            fence = None
        else:
            body.append(line)
    if fence is not None:                               # an unclosed fence runs to the end of the page
        fences.append((fence[1], fence[2], "\n".join(body)))
    return tuple(prose), tuple(fences)


def prose_lines(path: Path) -> tuple[tuple[int, str], ...]:
    return parsed(path)[0]


def code_spans(line: str) -> list[str]:
    return [match.group(2).strip() for match in CODE_SPAN.finditer(line)]


# --------------------------------------------------------------------------- links and anchors


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: the heading rendered to text (a link or image to its text, a code span to its
    content, an HTML tag and an emphasis marker outside code dropped), lower-cased, every character that is not a
    letter, digit, underscore, hyphen or space dropped, and each space a hyphen."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    parts, at = [], 0
    for match in CODE_SPAN.finditer(text):
        parts.append(_markup_dropped(text[at:match.start()]))
        parts.append(match.group(2))
        at = match.end()
    parts.append(_markup_dropped(text[at:]))
    text = "".join(parts).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _markup_dropped(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"(?<!\w)_+|_+(?!\w)", "", text)     # `_emphasis_`, but not the underscore inside a word


@functools.lru_cache(maxsize=None)
def anchors(path: Path) -> frozenset[str]:
    """Every anchor a page has: each heading's slug, a repeat numbered `-1`, `-2`, ..., and each `<a id>`."""
    seen: dict[str, int] = {}
    found: set[str] = set()
    for _, line in prose_lines(path):
        found.update(HTML_ANCHOR.findall(line))
        heading = HEADING.match(line)
        if heading is None:
            continue
        base = slug(heading.group(2))
        count = seen.get(base, 0)
        seen[base] = count + 1
        found.add(base if count == 0 else f"{base}-{count}")
    return frozenset(found)


def test_links_and_anchors_resolve():
    broken = []
    for path in doc_files():
        for number, line in prose_lines(path):
            for target in LINK.findall(CODE_SPAN.sub("code", line)):
                if re.match(r"^[a-z][a-z0-9+.\-]*:", target):
                    continue                            # http:, https:, mailto:
                file_part, _, anchor = target.partition("#")
                dest = (path.parent / file_part).resolve() if file_part else path
                if not dest.is_relative_to(ROOT):
                    broken.append(f"{rel(path)}:{number} {target} (outside the repo)")
                elif not dest.exists():
                    broken.append(f"{rel(path)}:{number} {target} (no such file)")
                elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                    broken.append(f"{rel(path)}:{number} {target} (no such anchor)")
    assert not broken, "broken links:\n" + "\n".join(broken)


# --------------------------------------------------------------------------- front matter


def front_matter(path: Path) -> dict[str, Any] | None:
    text = path.read_text()
    if not text.startswith("---\n"):
        return None
    block, sep, _ = text[4:].partition("\n---\n")
    if not sep:
        return None
    loaded = yaml.safe_load(block)
    return loaded if isinstance(loaded, dict) else None


def test_every_page_has_front_matter_naming_files_that_exist():
    wrong = []
    for path in sorted((ROOT / "docs").rglob("*.md")):
        try:
            fields = front_matter(path)
        except yaml.YAMLError as e:
            wrong.append(f"{rel(path)}: front matter is not YAML ({str(e).splitlines()[0]}); quote a value that "
                         "holds ': '")
            continue
        if fields is None:
            wrong.append(f"{rel(path)}: no front matter (a --- block first)")
            continue
        for key, kind in (("title", str), ("one_liner", str), ("tags", list)):
            if not isinstance(fields.get(key), kind):
                wrong.append(f"{rel(path)}: front matter has no {key} ({kind.__name__})")
        for key in ("related", "sources"):
            listed = fields.get(key) or []
            if not isinstance(listed, list):
                wrong.append(f"{rel(path)}: {key} is not a list")
                continue
            wrong += [f"{rel(path)} {key}: {entry}" for entry in listed if not (ROOT / str(entry)).exists()]
    assert not wrong, "front matter:\n" + "\n".join(wrong)


# --------------------------------------------------------------------------- paths and symbols


def located(token: str, *within: str) -> Path | None:
    """The repo file or directory `token` names: from the root, else from each of `within`."""
    for base in ("", *within):
        candidate = ROOT / base / token
        if candidate.exists():
            return candidate
    return None


def cited_paths(path: Path):
    """(line, token, the directories it may also be read from) for each repo path a page names in backticks."""
    for number, line in prose_lines(path):
        for span in code_spans(line):
            token = re.sub(r":\d+(?:-\d+)?$", "", span).rstrip(",.;")
            if token.startswith(REPO_DIRS) and (FILE_TOKEN.match(token) or DIR_TOKEN.match(token)):
                yield number, token, ()
            elif ROOT_FILE.match(token):
                yield number, token, ()
            elif (token.endswith(".py") and "/" in token and FILE_TOKEN.match(token)
                  and not token.startswith((*EXTERNAL, ".", "/"))):
                yield number, token, ("clab",)


def test_cited_paths_exist():
    missing = [f"{rel(path)}:{number} `{token}`"
               for path in doc_files()
               for number, token, within in cited_paths(path)
               if located(token, *within) is None]
    assert not missing, "cited paths that do not exist:\n" + "\n".join(missing)


@functools.lru_cache(maxsize=None)
def defines(file: Path, name: str) -> bool:
    text = file.read_text(errors="replace")
    name = re.escape(name)
    return bool(
        re.search(rf"^\s*(?:async\s+)?(?:def|class)\s+{name}\b", text, re.M)
        or re.search(rf"^\s*(?:self\.|cls\.)?{name}\s*[:=]", text, re.M)
        or re.search(rf"^\s*(?:from\s+[\w.]+\s+)?import\s+[^\n]*\b{name}\b", text, re.M)
        or re.search(rf"^from\s+[\w.]+\s+import\s+\([^)]*\b{name}\b", text, re.M)
    )


def symbol_citations(path: Path):
    """(line, path, symbol) for each symbol a page cites beside its `.py` file."""
    for number, line in prose_lines(path):
        pairs = [(m.group(1), m.group(2)) for m in PATH_THEN_SYMBOL.finditer(line)]
        pairs += [(m.group(2), m.group(1)) for m in SYMBOL_THEN_PATH.finditer(line)]
        for cited, symbol in pairs:
            if ":" in cited or FILE_TOKEN.match(symbol.removesuffix("()")):
                continue                                # a line-number citation, or two paths side by side
            if not cited.startswith(EXTERNAL):
                yield number, cited, symbol


def test_cited_symbols_are_defined():
    wrong = []
    for path in doc_files():
        for number, cited, symbol in symbol_citations(path):
            file = located(cited, "clab", "tests")
            if file is None or not file.is_file():
                wrong.append(f"{rel(path)}:{number} `{cited}` `{symbol}` (no such file)")
                continue
            parts = symbol.removesuffix("()").split(".")
            # `Class.method` needs both; a dotted module prefix (`plan.Plan.compile`) is skipped.
            if not all(defines(file, part) for part in parts[-2:]):
                wrong.append(f"{rel(path)}:{number} `{cited}` `{symbol}`")
    assert not wrong, "symbols not defined in the cited file:\n" + "\n".join(wrong)


def test_no_line_number_citations():
    found = [f"{rel(path)}:{number}"
             for path in doc_files()
             for number, line in prose_lines(path)
             for _ in LINE_CITATION.finditer(line)]
    assert len(found) <= LINE_CITATION_BASELINE, (
        f"{len(found)} line-number citations, baseline {LINE_CITATION_BASELINE}: cite path + symbol instead "
        f"(`clab/compile/positions.py` `landing`). {found[:10]}")


# --------------------------------------------------------------------------- banned terms


def test_no_banned_terms():
    patterns = [re.compile(term, re.I) for term in BANNED]
    found = [f"{rel(path)}:{number} {match.group(0)!r}"
             for path in doc_files()
             for number, line in prose_lines(path)
             for pattern in patterns
             for match in pattern.finditer(line)]
    assert not found, ("environment-specific or temporary names (generalize the requirement, or leave it out):\n"
                       + "\n".join(found))


# --------------------------------------------------------------------------- complete documents compile


def complete_documents() -> list[tuple[str, str]]:
    """(page:line of the opening fence, text) for each ```json block that holds a complete document, or tries to."""
    found = []
    for path in doc_files():
        for number, info, text in parsed(path)[1]:
            if info.lower() != "json":
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                complete = bool(COMPLETE.search(text))
            else:
                complete = isinstance(value, dict) and "model" in value and "steps" in value
            if complete:
                found.append((f"{rel(path)}:{number}", text))
    return found


COMPLETE_DOCUMENTS = complete_documents()


@pytest.mark.parametrize("where, text", COMPLETE_DOCUMENTS, ids=[where for where, _ in COMPLETE_DOCUMENTS])
def test_every_complete_document_compiles(where: str, text: str, corpus: str):
    error = None
    try:
        document = json.loads(text)
    except json.JSONDecodeError as e:
        document, error = None, (f"a ```json block with `model` and `steps` is a complete document, and this one is "
                                 f"not valid JSON ({e}). Complete it, or drop `model` to make it a fragment.")
    if document is not None:
        with tempfile.TemporaryDirectory(dir=corpus) as here:
            file = os.path.join(here, "document.json")
            with open(file, "w") as f:
                json.dump(document, f)
            try:
                Plan(file).compile()
            except Exception as e:
                error = f"the document does not compile: {type(e).__name__}: {e}"
    if error is not None:
        pytest.fail(f"{where}: {error}", pytrace=False)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
