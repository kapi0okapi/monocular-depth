"""Markdown tables of depth evaluations, next to the published numbers.

depth-summary --runs runs/benchmark runs/finetuned --out runs/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASSES = ("road", "car", "pole", "fence", "traffic sign", "vegetation", "building")


def load(dirs: list[Path]) -> list[dict]:
    return [json.loads(p.read_text()) for d in dirs for p in sorted(d.glob("*.json"))]


def label(report: dict) -> str:
    model = report["model"]
    return f"{Path(model).parent.parent.name} (fine-tuned)" if model.endswith(".pt") else model


def fmt(value, digits=3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def main_table(reports: list[dict]) -> str:
    lines = [
        "| Model | Output | Alignment | AbsRel ↓ | RMSE (m) ↓ | δ1 ↑ | Paper: AbsRel / δ1 | s/image |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(reports, key=lambda r: r["abs_rel"]):
        paper = r.get("paper")
        paper_cell = f"{fmt(paper[0])} / {fmt(paper[1])} ({paper[2]})" if paper else "–"
        lines.append(
            f"| {label(r)} | {r['output']} | {r['alignment']} | {fmt(r['abs_rel'])} | {fmt(r['rmse'], 2)} | "
            f"{fmt(r['delta1'])} | {paper_cell} | {fmt(r['seconds_per_image'], 2)} |"
        )
    return "\n".join(lines)


def class_table(reports: list[dict]) -> str:
    rows = [r for r in reports if r.get("abs_rel_by_class")]
    if not rows:
        return ""
    lines = ["| Model | " + " | ".join(CLASSES) + " |", "|---" * (len(CLASSES) + 1) + "|"]
    for r in sorted(rows, key=lambda r: r["abs_rel"]):
        cells = [fmt(r["abs_rel_by_class"].get(c)) for c in CLASSES]
        lines.append(f"| {label(r)} | " + " | ".join(cells) + " |")
    return "AbsRel by object class (pixels labelled by YOLO26s-sem):\n\n" + "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    reports = load(args.runs)
    text = main_table(reports) + "\n\n" + class_table(reports)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
