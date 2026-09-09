"""Check Python syntax, notebook structure, and local Markdown links without executing analysis."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import mistune
import nbformat


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = mistune.create_markdown(renderer="ast")


def walk(tokens):
    for token in tokens:
        yield token
        yield from walk(token.get("children", []))


def heading_ids(tokens):
    """GitHub-style identifiers for the ordinary Markdown headings used here."""
    used = set()
    for token in tokens:
        if token["type"] != "heading":
            continue
        text = "".join(t.get("raw", "") for t in walk(token.get("children", [])))
        base = re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")
        slug, suffix = base, 0
        while slug in used:
            suffix += 1
            slug = f"{base}-{suffix}"
        used.add(slug)
    return used


def check_links(path):
    tokens = MARKDOWN(path.read_text(encoding="utf-8"))
    for token in walk(tokens):
        if token["type"] not in {"link", "image"}:
            continue
        target = token["attrs"]["url"]
        url = urlsplit(target)
        if url.scheme or url.netloc:
            continue  # External links are outside this offline check.
        destination = (path.parent / unquote(url.path)).resolve() if url.path else path
        if not destination.is_relative_to(ROOT) or not destination.exists():
            raise ValueError(f"{path.relative_to(ROOT)}: missing local target {target}")
        fragment = unquote(url.fragment)
        if fragment and destination.suffix == ".md":
            if fragment not in heading_ids(MARKDOWN(destination.read_text(encoding="utf-8"))):
                raise ValueError(f"{path.relative_to(ROOT)}: missing heading {target}")
        elif fragment and re.fullmatch(r"L\d+(?:-L\d+)?", fragment):
            lines = [int(n) for n in re.findall(r"\d+", fragment)]
            if min(lines) < 1 or max(lines) > len(destination.read_text(encoding="utf-8").splitlines()):
                raise ValueError(f"{path.relative_to(ROOT)}: invalid line link {target}")


def main():
    sources = sorted(ROOT.glob("*.py")) + sorted((ROOT / "scripts").rglob("*.py"))
    for path in sources:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    notebooks = sorted(ROOT.glob("*.ipynb"))
    if not notebooks:
        raise ValueError("No renderable notebook found in the repository root.")
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells, 1):
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:cell-{index}", "exec")
    documents = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").rglob("*.md"))
    for path in documents:
        check_links(path)
    print(f"Validated {len(sources)} Python files, {len(notebooks)} notebooks, and {len(documents)} Markdown documents.")
    print("Historical notebook outputs were not executed or evaluated as model results.")


if __name__ == "__main__":
    main()
