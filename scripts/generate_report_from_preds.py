#!/usr/bin/env python3
"""
Generate a CSV/HTML report from evaluator predictions.

Usage example:
  python scripts/generate_report_from_preds.py \
    --preds checkpoints/evaluator_SudokuEvaluator_step_0/submission_preds.npy \
    --data-root data/sudoku-extreme-1k-aug-1000 \
    --out checkpoints/pred_report.csv \
    --html checkpoints/pred_report.html \
    --max 200

If your evaluator produced `submission.json` instead, point `--preds` to it;
this script will load it if `*.npy` is not found.
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import csv
from typing import Optional


def load_preds(path: str):
    if path.endswith(".npy") and os.path.exists(path):
        return np.load(path)
    if path.endswith(".json") and os.path.exists(path):
        with open(path, "r") as f:
            j = json.load(f)
        # j may be dict[str]->list; convert to array sorted by index if keys are numeric strings
        keys = sorted(j.keys(), key=lambda k: int(k))
        arr = np.array([j[k] for k in keys], dtype=np.int64)
        return arr
    raise FileNotFoundError(f"Preds file not found: {path}")


def grid_to_str(grid: np.ndarray) -> str:
    # grid: 9x9 ints
    rows = [" ".join(str(int(x)) for x in row) for row in grid]
    return "\\n".join(rows)


def grid_to_html_table(grid: np.ndarray, diff_mask: Optional[np.ndarray]=None) -> str:
    rows = []
    for r in range(grid.shape[0]):
        cells = []
        for c in range(grid.shape[1]):
            val = int(grid[r, c])
            style = ""
            if diff_mask is not None and diff_mask[r, c]:
                style = ' style="background:#ffcccc;"'  # highlight mismatches
            cells.append(f"<td{style}>{val}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table border='1' cellpadding='3' cellspacing='0'>" + "".join(rows) + "</table>"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True, help="Path to evaluator predictions (.npy or .json)")
    p.add_argument("--data-root", required=True, help="Dataset root (contains `test/` folder with all__inputs.npy etc.)")
    p.add_argument("--out", default="report.csv", help="CSV output path")
    p.add_argument("--html", default=None, help="Optional HTML output path")
    p.add_argument("--max", type=int, default=None, help="Limit number of examples for the report")
    p.add_argument("--start", type=int, default=0, help="Start index (useful for large datasets)")
    args = p.parse_args()

    preds = load_preds(args.preds)  # shape (N, seq_len)
    inputs = np.load(os.path.join(args.data_root, "test", "all__inputs.npy"), mmap_mode=None)
    labels = np.load(os.path.join(args.data_root, "test", "all__labels.npy"), mmap_mode=None)

    n = min(len(preds), len(inputs), len(labels))
    if args.max is not None:
        n = min(n, args.start + args.max)

    # tokens were saved as value+1 in dataset builder: convert back to digits by -1
    def to_digits(flat):
        arr = np.array(flat, dtype=np.int64).reshape(-1) - 1
        return arr.reshape(9, 9)

    # CSV header
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["index", "exact_match", "per_cell_accuracy", "input_grid", "label_grid", "pred_grid"])

        for i in range(args.start, n):
            inp_grid = to_digits(inputs[i])
            lab_grid = to_digits(labels[i])
            pred_grid = to_digits(preds[i])

            # compute metrics
            exact = bool(np.array_equal(pred_grid, lab_grid))
            per_cell_acc = float((pred_grid == lab_grid).mean())

            # flattened printable versions
            inp_str = "\\n".join(" ".join(str(int(x)) for x in row) for row in inp_grid)
            lab_str = "\\n".join(" ".join(str(int(x)) for x in row) for row in lab_grid)
            pred_str = "\\n".join(" ".join(str(int(x)) for x in row) for row in pred_grid)

            writer.writerow([i, exact, f"{per_cell_acc:.4f}", inp_str, lab_str, pred_str])

    print(f"Wrote CSV report to {args.out}")

    if args.html:
        # Simple HTML summary: for each example show three side-by-side tables, highlight mismatched cells
        os.makedirs(os.path.dirname(args.html) or '.', exist_ok=True)
        with open(args.html, "w") as hf:
            hf.write("<html><head><meta charset='utf-8'><title>Sudoku eval report</title></head><body>\n")
            hf.write("<h1>Sudoku predictions report</h1>\n")
            hf.write("<style>table{border-collapse:collapse;margin-right:20px;} td{width:28px;text-align:center;font-family:monospace}</style>\n")
            for i in range(args.start, n):
                inp_grid = to_digits(inputs[i])
                lab_grid = to_digits(labels[i])
                pred_grid = to_digits(preds[i])
                diff = (pred_grid != lab_grid)

                hf.write(f"<h3>Index {i} — exact={bool(np.array_equal(pred_grid, lab_grid))} — acc={float((pred_grid==lab_grid).mean()):.3f}</h3>\n")
                hf.write("<div style='display:flex;gap:20px;align-items:flex-start'>\n")
                hf.write("<div><b>Input</b><br/>" + grid_to_html_table(inp_grid) + "</div>\n")
                hf.write("<div><b>Label</b><br/>" + grid_to_html_table(lab_grid) + "</div>\n")
                hf.write("<div><b>Pred</b><br/>" + grid_to_html_table(pred_grid, diff_mask=diff) + "</div>\n")
                hf.write("</div><hr/>\n")
            hf.write("</body></html>\n")
        print(f"Wrote HTML report to {args.html}")


if __name__ == "__main__":
    main()
