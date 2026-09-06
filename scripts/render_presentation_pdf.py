from __future__ import annotations

import argparse
from pathlib import Path

import markdown


CSS = r"""
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body {
  font-family: "Segoe UI", Arial, sans-serif;
  color: #18212b;
  line-height: 1.45;
  font-size: 10.5pt;
}
h1 { color: #123b5d; font-size: 24pt; margin: 0 0 12pt; page-break-before: always; }
h1:first-child { page-break-before: auto; }
h2 { color: #155e75; font-size: 17pt; border-bottom: 1px solid #9bb8c8; padding-bottom: 4pt; margin-top: 20pt; }
h3 { color: #234e70; font-size: 12.5pt; margin-top: 14pt; }
p { margin: 6pt 0; }
blockquote { border-left: 4px solid #63a7b8; padding: 5pt 10pt; background: #eef7f8; }
pre { background: #f2f5f7; border: 1px solid #d6dee3; padding: 8pt; white-space: pre-wrap; font-family: Consolas, monospace; font-size: 8.5pt; }
code { font-family: Consolas, monospace; background: #f1f4f5; padding: 1pt 3pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 8.7pt; }
th { background: #155e75; color: white; font-weight: 600; }
th, td { border: 1px solid #aebdc5; padding: 5pt 6pt; vertical-align: top; }
tr:nth-child(even) td { background: #f5f9fa; }
li { margin: 3pt 0; }
hr { border: 0; border-top: 1px solid #ccd7dc; margin: 16pt 0; }
.math { font-family: Cambria Math, "Times New Roman", serif; font-size: 12pt; text-align: center; margin: 8pt 0; }
.note { background: #fff9e8; border-left: 4px solid #d79b27; padding: 6pt 10pt; }
"""


def render_markdown(source: Path, html_path: Path) -> None:
    text = source.read_text(encoding="utf-8")
    rendered = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "footnotes", "sane_lists"],
        output_format="html5",
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{source.stem}</title>
<style>{CSS}</style></head><body>{rendered}</body></html>"""
    html_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Markdown presentation guide to HTML for PDF printing.")
    parser.add_argument("source", type=Path)
    parser.add_argument("html", type=Path)
    args = parser.parse_args()
    args.html.parent.mkdir(parents=True, exist_ok=True)
    render_markdown(args.source, args.html)


if __name__ == "__main__":
    main()
