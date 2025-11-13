# Sudoku Solving Algorithm Analysis - Key Findings

## 🎯 Primary Strategy: GUESS_AND_CHECK / ITERATIVE REFINEMENT (Hybrid)

Based on analysis of 16 intermediate solving steps across 10 puzzles:

## Key Observations

### 1. **NOT Pure Backtracking**
- ✅ **0% rollback rate** - No cells ever go from filled → empty
- ✅ **0 total erases** - Cells are never blanked out once filled
- ❌ **178 cells with value flips** - BUT these are corrections, not backtracking

**Interpretation:** The model doesn't backtrack in the classical sense (undoing work). Instead, it **changes its mind** about cell values—this is iterative refinement.

### 2. **NOT Pure Monotonic Constraint Propagation**
- ❌ **453 monotonic violations** - Cells change value after being filled
- ❌ **Mean 0 new fills per step** - It's not progressively filling empty cells
- ✅ **42% cascade ratio** - Changes DO propagate through constraints

**Interpretation:** Not classical constraint propagation (fill empties with certain values). Instead, it's **refining existing guesses**.

### 3. **Iterative Refinement Pattern** ⭐
- ✅ **28.7% violation fix rate** - Nearly 1/3 of changes fix constraint violations
- ✅ **Cells change up to 15 times** - Gradual convergence
- ✅ **42% cascade ratio** - Changes respect Sudoku constraints (row/col/box)
- ✅ **No erases** - Always has a guess, just refines it

**Key Pattern:** The model starts with an initial guess for ALL cells (possibly based on embedded input), then iteratively refines cells that violate constraints or have low confidence.

### 4. **Spatial Clustering of Changes**
- 55 cascade steps (changes in same row/col/box)
- 18 single-cell updates
- 58 independent multi-cell updates

**Interpretation:** When the model updates a cell, it often triggers related updates in the same constraint group—suggesting it's using **constraint awareness** but not pure logic.

## 🔬 Algorithm Characterization

### What the Model is Doing:

1. **Initial Phase:** Generate full-grid prediction (all 81 cells filled)
2. **Refinement Loop (16 steps):**
   - Identify cells that violate row/col/box constraints
   - Update those cells (and related cells in same constraint groups)
   - Repeat until convergence or step limit

### This is Most Similar To:
**Iterative Local Search / Constraint-guided Refinement**
- Like simulated annealing but with neural guidance
- Like WalkSAT but with learned heuristics instead of random walks
- Like human "guess-and-check" but systematic

### NOT Similar To:
- ❌ Backtracking (no tree search, no undo)
- ❌ Pure constraint propagation (changes filled cells, not just fills empties)
- ❌ Random search (changes respect constraints 28.7% of the time)

## 📊 Confidence Scores

Without logits, we can't measure **certainty**, but behavioral evidence suggests:

```
GUESS_AND_CHECK:          41.7%  (changes cells multiple times)
CONSTRAINT_PROPAGATION:   25.0%  (42% cascade ratio)
ITERATIVE_REFINEMENT:     16.7%  (28.7% fix rate)
BACKTRACKING:             16.7%  (many flips, but no erases)
```

## 🎓 Interpretation for Neural Approach

The TRM model appears to implement a **learned iterative refinement algorithm** where:

1. The transformer learns to **predict constraint violations**
2. Adaptive Computation Time (ACT) allows it to **take multiple refinement steps**
3. Each step **corrects mistakes** from previous steps
4. The process is **constraint-aware but not constraint-solving**

### Why This Works:

- **Efficient:** No backtracking means no exponential search
- **Parallelizable:** Can update multiple cells at once (cascade steps)
- **Learnable:** Neural net learns which cells to update and what values to try
- **Robust:** Multiple refinement steps allow recovery from bad initial guesses

### Limitations Without Logits:

⚠️ To fully confirm this hypothesis, we need:
- Entropy trajectory (are early guesses low-confidence?)
- Confidence at fill (do refined cells have higher confidence?)
- Logit changes (is the model "getting more sure" over time?)

## 🔄 Next Steps to Verify

1. **Capture logits** in intermediate results
2. **Re-run analysis** with `analyze_solving_algorithm.py`
3. **Check entropy trajectory** - should decrease over steps
4. **Visualize confidence heatmaps** - should increase over steps for changing cells

---

Generated: 2025-11-13
Tool: `scripts/analyze_solving_algorithm.py`
Data: 16 steps × 10 puzzles = 160 puzzle-solving trajectories
