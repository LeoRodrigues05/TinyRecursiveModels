#!/usr/bin/env python3
"""
Generate HTML report with intermediate results and colored cells.
Shows step-by-step evolution of Sudoku solving with validation coloring.

Usage:
    python3 scripts/generate_html_report_with_intermediates.py [output.html]
    
Or customize paths in the script directly.
"""

import sys
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# ---- CONFIG ----
OUT_HTML = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/eval_10examples/sudoku_eval_report_colored.html"
PRED_PATH = "checkpoints/eval_10examples/evaluator_SudokuEvaluator_step_100/submission_preds.npy"
INPUTS_PATH = "data/sudoku-extreme-1k-aug-1000-25pct/test/all__inputs.npy"
LABELS_PATH = "data/sudoku-extreme-1k-aug-1000-25pct/test/all__labels.npy"
INTERM_PATH = "checkpoints/eval_10examples/intermediate_results/step_100_batch_1.pkl"
NUM_EXAMPLES = 10

# ---- HELPERS ----
def id2num(i: int) -> str:
    """Convert token ID to Sudoku digit string (2-10 -> '1'-'9', else '.')"""
    if 2 <= i <= 10:
        return str(i - 1)
    return "."

def to_rows(vec81):
    """Split 81-length vector into 9 rows of 9."""
    return [vec81[r * 9 : (r + 1) * 9] for r in range(9)]

def table_html(title, arr81_chars, classes=None):
    """Generate HTML table for a 9x9 Sudoku grid with optional cell classes."""
    rows = to_rows(arr81_chars)
    h = [f"<div class='gridTitle'>{title}</div><table class='sgrid'>"]
    for r in range(9):
        h.append("<tr>")
        for c in range(9):
            ch = arr81_chars[r * 9 + c]
            cls = classes[r * 9 + c] if classes else ""
            borders = []
            # Thicker borders for 3x3 box boundaries
            if r in (2, 5):
                borders.append("bb")
            if c in (2, 5):
                borders.append("br")
            cls = (cls + " " + " ".join(borders)).strip()
            h.append(f"<td class='{cls}'>{ch}</td>")
        h.append("</tr>")
    h.append("</table>")
    return "\n".join(h)

# ---- Sudoku validation / styling ----
def compute_violations(chars81):
    """
    Return list of 81 bools indicating which filled cells violate 
    row/col/box uniqueness constraints.
    """
    viol = [False] * 81
    
    # Check rows
    for r in range(9):
        vals = [chars81[r*9+c] for c in range(9) if chars81[r*9+c] != "."]
        for c in range(9):
            ch = chars81[r*9+c]
            if ch != "." and vals.count(ch) > 1:
                viol[r*9+c] = True
    
    # Check columns
    for c in range(9):
        col_vals = [chars81[r*9+c] for r in range(9) if chars81[r*9+c] != "."]
        for r in range(9):
            ch = chars81[r*9+c]
            if ch != "." and col_vals.count(ch) > 1:
                viol[r*9+c] = True
    
    # Check 3x3 boxes
    for br in range(3):
        for bc in range(3):
            idxs = [(br*3+rr)*9 + (bc*3+cc) for rr in range(3) for cc in range(3)]
            box_vals = [chars81[i] for i in idxs if chars81[i] != "."]
            for i in idxs:
                ch = chars81[i]
                if ch != "." and box_vals.count(ch) > 1:
                    viol[i] = True
    
    return viol

def compute_classes(curr_chars, given_chars, prev_chars=None):
    """
    Compute CSS class for each cell based on:
    - Priority: changed (half yellow) > given (blue) > ok/bad > blank
    - changed_ok: cell changed from previous step and is valid
    - changed_bad: cell changed from previous step but violates constraints
    - given: initial puzzle clue (blue background)
    - ok: valid filled cell (green)
    - bad: invalid filled cell (red)
    - blank: empty cell
    """
    viol = compute_violations(curr_chars)
    classes = []
    for i in range(81):
        ch = curr_chars[i]
        given = (given_chars[i] != ".")
        changed = (prev_chars is not None and ch != prev_chars[i] and ch != ".")
        
        if changed:
            # Changed cells get diagonal yellow/green or yellow/red
            cls = "changed_ok" if not viol[i] else "changed_bad"
        elif given:
            # Given clues are blue
            cls = "given"
        elif ch == ".":
            # Empty cells
            cls = "blank"
        else:
            # Filled cells: green if valid, red if invalid
            cls = "ok" if not viol[i] else "bad"
        classes.append(cls)
    return classes

def render_one(i, inputs, labels, pred, interm_steps=None):
    """Render HTML for one puzzle with input, prediction, label, and intermediate steps."""
    givens       = [id2num(int(x)) for x in inputs[i]]
    label_chars  = [id2num(int(x)) for x in labels[i]]
    pred_chars   = [id2num(int(x)) for x in pred[i]]

    pred_classes = compute_classes(pred_chars, givens, prev_chars=None)

    sec = []
    sec.append(f"<h3>Puzzle {i+1}</h3>")
    sec.append("<div class='row3'>")
    sec.append(table_html("Input",   givens))
    sec.append(table_html("Prediction", pred_chars, pred_classes))
    sec.append(table_html("Label",   label_chars))
    sec.append("</div>")

    # Intermediate steps with change tracking and validity coloring
    if interm_steps is not None and len(interm_steps) > 0:
        S = len(interm_steps)
        sec.append("<details><summary>Show intermediate steps</summary>")
        sec.append("<div class='steps'>")
        prev_chars = None
        for s in range(S):
            step = interm_steps[s][i]
            # If logits (shape: 81 x 10), take argmax
            if step.ndim == 2:
                step = step.argmax(-1)
            step_chars = [id2num(int(x)) for x in (step if step.ndim == 1 else step.squeeze())]
            classes = compute_classes(step_chars, givens, prev_chars=prev_chars)
            sec.append(table_html(f"Step {s+1}/{S}", step_chars, classes))
            prev_chars = step_chars
        sec.append("</div></details>")
    return "\n".join(sec)

# ---- MAIN ----
def main():
    print(f"Loading data...")
    print(f"  Inputs: {INPUTS_PATH}")
    print(f"  Labels: {LABELS_PATH}")
    print(f"  Predictions: {PRED_PATH}")
    print(f"  Intermediate: {INTERM_PATH}")
    
    # Load data
    inputs = np.load(INPUTS_PATH)
    labels = np.load(LABELS_PATH)
    pred   = np.load(PRED_PATH)
    
    # Load intermediate results (list of dicts, one per step)
    try:
        with open(INTERM_PATH, "rb") as f:
            interm = pickle.load(f)
        # Each step: interm[s]["predictions"] shape (batch, 81)
        interm_steps = [step["predictions"].numpy() if hasattr(step["predictions"], 'numpy') else step["predictions"] 
                       for step in interm]
        print(f"  Loaded {len(interm_steps)} intermediate steps")
    except FileNotFoundError:
        print(f"  Warning: Intermediate file not found, skipping intermediate steps")
        interm_steps = None

    N = min(NUM_EXAMPLES, len(pred))
    print(f"\nGenerating report for {N} examples...")
    sections = [render_one(i, inputs, labels, pred, interm_steps=interm_steps) for i in range(N)]

    # CSS with darker colors and diagonal split for changed cells
    css = """
    <style>
    body { 
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; 
        background:#fafafa; 
        color:#222; 
        margin: 20px;
    }
    h1 { margin: 8px 0 0 0; }
    .meta { color:#555; margin-bottom: 16px; font-size: 14px; }
    .row3 { 
        display:grid; 
        grid-template-columns: repeat(3, max-content); 
        gap:20px; 
        align-items:start; 
        margin-bottom: 20px;
    }
    .sgrid { 
        border-collapse:collapse; 
        margin:6px 0 12px 0; 
    }
    .sgrid td { 
        width:22px; 
        height:22px; 
        text-align:center; 
        border:1px solid #777; 
        padding:2px 4px; 
        position:relative;
        font-size: 14px;
        font-weight: 500;
    }
    .sgrid td.br { border-right:2px solid #111; }
    .sgrid td.bb { border-bottom:2px solid #111; }
    .gridTitle { 
        font-weight:700; 
        margin:4px 0; 
        font-size: 13px;
        color: #333;
    }

    /* Darker, more distinct fills */
    .given   { background:#9AB0FF; font-weight:700; }  /* darker blue for given clues */
    .ok      { background:#57B97B; color:#101; }       /* darker green for valid fills */
    .bad     { background:#E86B6B; color:#101; }       /* darker red for invalid fills */
    .blank   { color:#667; }                           /* gray for empty cells */

    /* Changed cells: diagonal split yellow/green or yellow/red */
    /* These indicate cells that changed from the previous step */
    .changed_ok {
      background-image: linear-gradient(135deg, #FFE15A 50%, #57B97B 50%);
      background-color: #57B97B;  /* fallback */
      font-weight: 600;
    }
    .changed_bad {
      background-image: linear-gradient(135deg, #FFE15A 50%, #E86B6B 50%);
      background-color: #E86B6B;  /* fallback */
      font-weight: 600;
    }

    details { 
        margin-top: 6px; 
        border: 1px solid #ddd;
        padding: 8px;
        border-radius: 4px;
        background: #fff;
    }
    summary {
        cursor: pointer;
        font-weight: 600;
        color: #0066cc;
        padding: 4px;
    }
    summary:hover {
        color: #004499;
    }
    .steps { 
        display:grid; 
        grid-template-columns: repeat(5, max-content); 
        gap:18px; 
        margin-top: 12px;
    }
    
    h3 {
        margin-top: 30px;
        padding-bottom: 8px;
        border-bottom: 2px solid #ddd;
    }
    </style>
    """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sudoku Eval Report</title>{css}</head>
<body>
<h1>Sudoku Evaluation Report</h1>
<div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
Predictions: {PRED_PATH} | 
Intermediate: {INTERM_PATH}</div>
<div class="meta"><strong>Color Legend:</strong> 
<span style="background:#9AB0FF; padding:2px 6px;">Given</span> 
<span style="background:#57B97B; padding:2px 6px;">Valid</span> 
<span style="background:#E86B6B; padding:2px 6px;">Invalid</span> 
<span style="background:linear-gradient(135deg, #FFE15A 50%, #57B97B 50%); padding:2px 6px;">Changed (Valid)</span>
<span style="background:linear-gradient(135deg, #FFE15A 50%, #E86B6B 50%); padding:2px 6px;">Changed (Invalid)</span>
</div>
{''.join(sections)}
</body></html>
"""
    Path(OUT_HTML).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_HTML).write_text(html, encoding="utf-8")
    print(f"\n✓ Wrote {OUT_HTML}")
    print(f"  Open it in your browser to view the report.")

if __name__ == "__main__":
    main()
