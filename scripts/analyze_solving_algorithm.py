#!/usr/bin/env python3
"""
Comprehensive Analysis: Identify Sudoku Solving Algorithm

Analyzes intermediate solving steps to determine what algorithm the model employs:
- Backtracking
- Constraint Propagation
- Iterative Refinement
- Guess-and-Check
- Hybrid approach

Usage:
    python3 scripts/analyze_solving_algorithm.py \
        --intermediate checkpoints/eval_10examples/intermediate_results/step_100_batch_1.pkl \
        --inputs data/sudoku-extreme-1k-aug-1000-25pct/test/all__inputs.npy \
        --output results/algorithm_analysis/
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import entropy
import json

# ============================================================================
# TOKEN MAPPING (matching your conventions)
# ============================================================================

def id2num(i: int) -> str:
    """Convert token ID to Sudoku digit string (2-10 -> '1'-'9', else '.')"""
    if 2 <= i <= 10:
        return str(i - 1)
    return "."

def to_chars(arr_ids: np.ndarray) -> np.ndarray:
    """Convert array of token IDs to character strings"""
    vfunc = np.vectorize(id2num)
    return vfunc(arr_ids.astype(int))

def safe_log_softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    """Numerically stable log softmax"""
    x = x - np.max(x, axis=axis, keepdims=True)
    logsum = np.log(np.sum(np.exp(x), axis=axis, keepdims=True) + 1e-12)
    return x - logsum

# ============================================================================
# CONSTRAINT VIOLATION CHECKING
# ============================================================================

def compute_violations_per_cell(chars_batch: np.ndarray) -> np.ndarray:
    """
    Check which cells violate Sudoku constraints.
    
    Args:
        chars_batch: [N, 81] array of characters
        
    Returns:
        [N, 81] boolean array where True = cell violates constraint
    """
    N = chars_batch.shape[0]
    violations = np.zeros((N, 81), dtype=bool)
    
    for i in range(N):
        chars81 = chars_batch[i]
        
        # Check rows
        for r in range(9):
            row = chars81[r*9:(r+1)*9]
            vals = row[row != '.']
            if len(vals) != len(set(vals.tolist())):  # duplicates
                for c in range(9):
                    idx = r*9 + c
                    if chars81[idx] != '.' and np.sum(row == chars81[idx]) > 1:
                        violations[i, idx] = True
        
        # Check columns
        for c in range(9):
            col = chars81[c::9]
            vals = col[col != '.']
            if len(vals) != len(set(vals.tolist())):
                for r in range(9):
                    idx = r*9 + c
                    if chars81[idx] != '.' and np.sum(col == chars81[idx]) > 1:
                        violations[i, idx] = True
        
        # Check 3x3 boxes
        for br in range(3):
            for bc in range(3):
                box_indices = [(br*3+rr)*9 + (bc*3+cc) for rr in range(3) for cc in range(3)]
                box = chars81[box_indices]
                vals = box[box != '.']
                if len(vals) != len(set(vals.tolist())):
                    for idx in box_indices:
                        if chars81[idx] != '.' and np.sum(box == chars81[idx]) > 1:
                            violations[i, idx] = True
    
    return violations

# ============================================================================
# ANALYSIS 1: BACKTRACKING DETECTION
# ============================================================================

def detect_backtracking(steps_chars: np.ndarray, givens_mask: np.ndarray) -> dict:
    """
    Detect backtracking behavior:
    - Cells that go from filled → empty (rollback)
    - Cells that change value multiple times (flip)
    
    Args:
        steps_chars: [S, N, 81] character arrays
        givens_mask: [N, 81] boolean mask of given cells
        
    Returns:
        Dictionary with backtracking metrics
    """
    S, N, _ = steps_chars.shape
    
    rollback_steps = 0  # steps where any cell goes filled → empty
    cell_changes = np.zeros((N, 81), dtype=int)  # track changes per cell
    value_flips = np.zeros((N, 81), dtype=int)  # track value changes (not empty)
    
    for s in range(1, S):
        prev, curr = steps_chars[s-1], steps_chars[s]
        
        # Rollback: was filled, now empty (excluding givens)
        for i in range(N):
            was_filled = (prev[i] != '.') & ~givens_mask[i]
            now_empty = (curr[i] == '.')
            if np.any(was_filled & now_empty):
                rollback_steps += 1
                break
        
        # Track changes per cell
        for i in range(N):
            changed = (prev[i] != curr[i]) & ~givens_mask[i]
            cell_changes[i] += changed.astype(int)
            
            # Value flips (filled cell changes to different filled value)
            was_filled = (prev[i] != '.') & ~givens_mask[i]
            now_different = (curr[i] != prev[i]) & (curr[i] != '.')
            value_flips[i] += (was_filled & now_different).astype(int)
    
    return {
        'rollback_rate': rollback_steps / (S - 1) if S > 1 else 0,
        'rollback_steps': int(rollback_steps),
        'mean_changes_per_cell': float(cell_changes.mean()),
        'max_changes_per_cell': int(cell_changes.max()),
        'cells_with_multiple_changes': int((cell_changes > 1).sum()),
        'mean_value_flips_per_cell': float(value_flips.mean()),
        'max_value_flips': int(value_flips.max()),
        'cells_with_value_flips': int((value_flips > 0).sum())
    }

# ============================================================================
# ANALYSIS 2: MONOTONIC FILLING
# ============================================================================

def analyze_monotonicity(steps_chars: np.ndarray, givens_mask: np.ndarray) -> dict:
    """
    Check if solving is monotonic (cells only go empty → filled, never change value).
    
    Returns:
        Dictionary with monotonicity metrics
    """
    S, N, _ = steps_chars.shape
    
    monotonic_violations = 0  # times a filled cell changed value
    new_fills_per_step = []
    erases_per_step = []
    
    for s in range(1, S):
        prev, curr = steps_chars[s-1], steps_chars[s]
        
        step_violations = 0
        step_fills = 0
        step_erases = 0
        
        for i in range(N):
            mask = ~givens_mask[i]
            
            # Violation: cell had value, now has different value
            was_filled = (prev[i] != '.') & mask
            now_different = (curr[i] != prev[i]) & (curr[i] != '.')
            step_violations += np.sum(was_filled & now_different)
            
            # Count new fills (empty → filled)
            was_empty = (prev[i] == '.') & mask
            now_filled = (curr[i] != '.')
            step_fills += np.sum(was_empty & now_filled)
            
            # Count erases (filled → empty)
            was_filled = (prev[i] != '.') & mask
            now_empty = (curr[i] == '.')
            step_erases += np.sum(was_filled & now_empty)
        
        monotonic_violations += step_violations
        new_fills_per_step.append(step_fills)
        erases_per_step.append(step_erases)
    
    return {
        'is_monotonic': bool(monotonic_violations == 0),
        'monotonic_violations': int(monotonic_violations),
        'mean_new_fills_per_step': float(np.mean(new_fills_per_step)) if new_fills_per_step else 0,
        'std_new_fills_per_step': float(np.std(new_fills_per_step)) if new_fills_per_step else 0,
        'mean_erases_per_step': float(np.mean(erases_per_step)) if erases_per_step else 0,
        'total_erases': int(np.sum(erases_per_step))
    }

# ============================================================================
# ANALYSIS 3: ITERATIVE REFINEMENT
# ============================================================================

def analyze_refinement_pattern(steps_logits: np.ndarray, steps_chars: np.ndarray, 
                               givens_mask: np.ndarray) -> dict:
    """
    Detect iterative refinement by checking:
    1. Entropy decrease over time (increasing confidence)
    2. Small, localized changes per step
    3. Changes target constraint violations
    """
    S, N, cells, vocab = steps_logits.shape
    
    # 1. Entropy trajectory
    entropies = []
    max_probs = []
    for s in range(S):
        probs = np.exp(safe_log_softmax(steps_logits[s], axis=-1))
        # Only compute for non-given cells
        ent_per_cell = entropy(probs, axis=-1)  # [N, 81]
        max_prob_per_cell = probs.max(axis=-1)  # [N, 81]
        
        # Average over non-given cells only
        ent_values = []
        prob_values = []
        for i in range(N):
            mask = ~givens_mask[i]
            if mask.any():
                ent_values.append(ent_per_cell[i][mask].mean())
                prob_values.append(max_prob_per_cell[i][mask].mean())
        
        entropies.append(np.mean(ent_values) if ent_values else 0)
        max_probs.append(np.mean(prob_values) if prob_values else 0)
    
    # 2. Changes per step
    changes_per_step = []
    for s in range(1, S):
        changed = (steps_chars[s-1] != steps_chars[s])
        for i in range(N):
            changes_per_step.append((changed[i] & ~givens_mask[i]).sum())
    
    # 3. Violation targeting
    violation_fix_rate = compute_violation_targeting(steps_chars, givens_mask)
    
    # Check if entropy decreases
    entropy_diffs = np.diff(entropies)
    entropy_decreases = np.sum(entropy_diffs < 0) / len(entropy_diffs) if len(entropy_diffs) > 0 else 0
    
    return {
        'entropy_decrease_ratio': float(entropy_decreases),
        'entropy_start': float(entropies[0]) if entropies else 0,
        'entropy_end': float(entropies[-1]) if entropies else 0,
        'entropy_reduction': float(entropies[0] - entropies[-1]) if entropies else 0,
        'confidence_start': float(max_probs[0]) if max_probs else 0,
        'confidence_end': float(max_probs[-1]) if max_probs else 0,
        'mean_changes_per_step': float(np.mean(changes_per_step)) if changes_per_step else 0,
        'std_changes_per_step': float(np.std(changes_per_step)) if changes_per_step else 0,
        'violation_fix_rate': float(violation_fix_rate),
        'entropy_trajectory': [float(e) for e in entropies],
        'confidence_trajectory': [float(p) for p in max_probs]
    }

def compute_violation_targeting(steps_chars: np.ndarray, givens_mask: np.ndarray) -> float:
    """Fraction of changes that fix constraint violations"""
    S, N, _ = steps_chars.shape
    fixes = 0
    total_changes = 0
    
    for s in range(1, S):
        prev_viol = compute_violations_per_cell(steps_chars[s-1])
        curr_viol = compute_violations_per_cell(steps_chars[s])
        
        for i in range(N):
            changed = (steps_chars[s-1][i] != steps_chars[s][i]) & ~givens_mask[i]
            
            # Did changed cells reduce violations?
            fixes += np.sum(changed & prev_viol[i] & ~curr_viol[i])
            total_changes += np.sum(changed)
    
    return fixes / max(total_changes, 1)

# ============================================================================
# HAMMING DISTANCE ANALYSIS
# ============================================================================

def compute_hamming_distance(steps_chars: np.ndarray, givens_mask: np.ndarray) -> dict:
    """
    Compute step-to-step Hamming distance (fraction of cells changed).
    
    Returns:
        Dictionary with mean and std of Hamming distance per step
    """
    S, N, _ = steps_chars.shape
    
    hamming_per_step = []  # [S-1] - one value per step transition
    hamming_per_puzzle_per_step = []  # [S-1, N] - for std calculation
    
    for s in range(1, S):
        prev, curr = steps_chars[s-1], steps_chars[s]
        
        hamming_for_puzzles = []
        for i in range(N):
            mask = ~givens_mask[i]
            denom = int(mask.sum())
            
            if denom > 0:
                changed = (prev[i] != curr[i]) & mask
                hamming = float(changed.sum()) / denom
                hamming_for_puzzles.append(hamming)
            else:
                hamming_for_puzzles.append(0.0)
        
        hamming_per_puzzle_per_step.append(hamming_for_puzzles)
        hamming_per_step.append(np.mean(hamming_for_puzzles))
    
    hamming_per_puzzle_per_step = np.array(hamming_per_puzzle_per_step)  # [S-1, N]
    
    return {
        'mean_hamming_per_step': [float(h) for h in hamming_per_step],
        'std_hamming_per_step': [float(hamming_per_puzzle_per_step[s].std()) 
                                 for s in range(len(hamming_per_step))],
        'overall_mean_hamming': float(np.mean(hamming_per_step)),
        'overall_std_hamming': float(np.std(hamming_per_step))
    }

def compute_stepwise_kl(steps_logits: np.ndarray, givens_mask: np.ndarray) -> dict:
    """
    Compute KL divergence between consecutive steps' logit distributions.
    
    Returns:
        Dictionary with mean KL per step
    """
    S, N, cells, vocab = steps_logits.shape
    
    kl_per_step = []
    
    for s in range(1, S):
        l_prev = steps_logits[s-1]  # [N, 81, V]
        l_curr = steps_logits[s]    # [N, 81, V]
        
        # Log softmax
        lp = safe_log_softmax(l_prev, axis=-1)
        lq = safe_log_softmax(l_curr, axis=-1)
        p = np.exp(lp)
        
        # KL(p || q) per cell
        kl = (p * (lp - lq)).sum(axis=-1)  # [N, 81]
        
        # Average over non-given cells only
        kl_values = []
        for i in range(N):
            mask = ~givens_mask[i]
            if mask.any():
                kl_values.append(kl[i][mask].mean())
        
        kl_per_step.append(np.mean(kl_values) if kl_values else 0)
    
    return {
        'mean_kl_per_step': [float(k) for k in kl_per_step],
        'overall_mean_kl': float(np.mean(kl_per_step)) if kl_per_step else 0
    }

# ============================================================================
# ANALYSIS 4: CONSTRAINT PROPAGATION
# ============================================================================

def analyze_propagation_waves(steps_chars: np.ndarray, givens_mask: np.ndarray) -> dict:
    """
    Detect if changes propagate through constraints:
    - Fill one cell → triggers fills in same row/col/box
    """
    S, N, _ = steps_chars.shape
    
    cascade_steps = 0
    independent_fills = 0
    single_fills = 0
    
    for s in range(1, S):
        prev, curr = steps_chars[s-1], steps_chars[s]
        
        for i in range(N):
            changed_indices = np.where((prev[i] != curr[i]) & ~givens_mask[i])[0]
            
            if len(changed_indices) == 1:
                single_fills += 1
            elif len(changed_indices) > 1:
                independent_fills += 1
                
                # Check if changes are in same row/col/box
                rows = changed_indices // 9
                cols = changed_indices % 9
                boxes = (rows // 3) * 3 + (cols // 3)
                
                # If multiple cells in same constraint, likely a cascade
                if len(np.unique(rows)) < len(changed_indices) or \
                   len(np.unique(cols)) < len(changed_indices) or \
                   len(np.unique(boxes)) < len(changed_indices):
                    cascade_steps += 1
    
    total = cascade_steps + independent_fills + single_fills
    
    return {
        'cascade_steps': int(cascade_steps),
        'independent_multi_fills': int(independent_fills),
        'single_fills': int(single_fills),
        'cascade_ratio': float(cascade_steps / max(total, 1)),
        'single_fill_ratio': float(single_fills / max(total, 1))
    }

# ============================================================================
# ANALYSIS 5: CERTAINTY AT FILL
# ============================================================================

def analyze_certainty_at_fill(steps_logits: np.ndarray, steps_chars: np.ndarray,
                               givens_mask: np.ndarray) -> dict:
    """
    When cells are first filled, how confident is the model?
    - High confidence → logical deduction
    - Low confidence → guessing
    """
    S, N, cells, vocab = steps_logits.shape
    
    fill_confidences = []
    
    for s in range(1, S):
        prev, curr = steps_chars[s-1], steps_chars[s]
        
        probs = np.exp(safe_log_softmax(steps_logits[s], axis=-1))
        max_probs = probs.max(axis=-1)  # [N, 81]
        
        for i in range(N):
            # Newly filled cells (was empty, now filled)
            newly_filled = (prev[i] == '.') & (curr[i] != '.') & ~givens_mask[i]
            
            if np.any(newly_filled):
                fill_confidences.extend(max_probs[i][newly_filled].tolist())
    
    if not fill_confidences:
        return {
            'mean_fill_confidence': 0,
            'median_fill_confidence': 0,
            'min_fill_confidence': 0,
            'low_confidence_fills': 0,
            'high_confidence_fills': 0
        }
    
    fill_confidences = np.array(fill_confidences)
    
    return {
        'mean_fill_confidence': float(fill_confidences.mean()),
        'median_fill_confidence': float(np.median(fill_confidences)),
        'min_fill_confidence': float(fill_confidences.min()),
        'max_fill_confidence': float(fill_confidences.max()),
        'low_confidence_fills': int(np.sum(fill_confidences < 0.6)),
        'high_confidence_fills': int(np.sum(fill_confidences > 0.9)),
        'confidence_distribution': {
            '0.0-0.2': int(np.sum(fill_confidences < 0.2)),
            '0.2-0.4': int(np.sum((fill_confidences >= 0.2) & (fill_confidences < 0.4))),
            '0.4-0.6': int(np.sum((fill_confidences >= 0.4) & (fill_confidences < 0.6))),
            '0.6-0.8': int(np.sum((fill_confidences >= 0.6) & (fill_confidences < 0.8))),
            '0.8-1.0': int(np.sum(fill_confidences >= 0.8))
        }
    }

# ============================================================================
# ALGORITHM CLASSIFICATION
# ============================================================================

def classify_algorithm(results: dict) -> dict:
    """
    Classify the solving strategy based on all metrics.
    
    Returns:
        Dictionary with primary strategy and confidence scores
    """
    scores = {
        'BACKTRACKING': 0,
        'CONSTRAINT_PROPAGATION': 0,
        'ITERATIVE_REFINEMENT': 0,
        'GUESS_AND_CHECK': 0
    }
    
    bt = results['backtracking']
    mono = results['monotonic']
    refine = results['refinement']
    prop = results['propagation']
    cert = results['certainty']
    
    # Backtracking indicators
    if bt['rollback_rate'] > 0.1:
        scores['BACKTRACKING'] += 40
    if bt['rollback_rate'] > 0.05:
        scores['BACKTRACKING'] += 20
    if bt['cells_with_value_flips'] > 10:
        scores['BACKTRACKING'] += 20
    if mono['total_erases'] > 0:
        scores['BACKTRACKING'] += 10
    
    # Constraint propagation indicators
    if mono['is_monotonic']:
        scores['CONSTRAINT_PROPAGATION'] += 30
    if prop['cascade_ratio'] > 0.3:
        scores['CONSTRAINT_PROPAGATION'] += 30
    if prop['single_fill_ratio'] > 0.5:
        scores['CONSTRAINT_PROPAGATION'] += 20
    if cert['mean_fill_confidence'] > 0.8:
        scores['CONSTRAINT_PROPAGATION'] += 20
    
    # Iterative refinement indicators
    if refine['entropy_decrease_ratio'] > 0.7:
        scores['ITERATIVE_REFINEMENT'] += 30
    if refine['violation_fix_rate'] > 0.6:
        scores['ITERATIVE_REFINEMENT'] += 30
    if not mono['is_monotonic'] and mono['monotonic_violations'] > 0:
        scores['ITERATIVE_REFINEMENT'] += 20
    if refine['confidence_end'] > refine['confidence_start'] + 0.1:
        scores['ITERATIVE_REFINEMENT'] += 20
    
    # Guess-and-check indicators
    if cert['mean_fill_confidence'] < 0.6:
        scores['GUESS_AND_CHECK'] += 30
    if cert['low_confidence_fills'] > cert['high_confidence_fills']:
        scores['GUESS_AND_CHECK'] += 30
    if refine['entropy_decrease_ratio'] < 0.3:
        scores['GUESS_AND_CHECK'] += 20
    
    # Normalize scores
    total = sum(scores.values())
    if total > 0:
        normalized_scores = {k: v/total for k, v in scores.items()}
    else:
        normalized_scores = scores
    
    primary_strategy = max(scores.items(), key=lambda x: x[1])[0]
    
    # Determine if hybrid
    top_two = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
    is_hybrid = (top_two[0][1] - top_two[1][1]) < 20
    
    return {
        'primary_strategy': primary_strategy,
        'is_hybrid': bool(is_hybrid),
        'confidence_scores': normalized_scores,
        'raw_scores': scores,
        'secondary_strategy': top_two[1][0] if is_hybrid else None
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(results: dict, output_dir: Path):
    """Create plots for key metrics"""
    
    # 1. Hamming distance plot (matching your sample script)
    if 'hamming' in results and results['hamming']['mean_hamming_per_step']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mean_ham = results['hamming']['mean_hamming_per_step']
        std_ham = results['hamming']['std_hamming_per_step']
        steps = list(range(1, len(mean_ham) + 1))
        
        ax.plot(steps, mean_ham, 'b-', linewidth=2, label='Mean Hamming Δ')
        ax.fill_between(steps, 
                        [m - s for m, s in zip(mean_ham, std_ham)],
                        [m + s for m, s in zip(mean_ham, std_ham)],
                        alpha=0.25, label='±1σ')
        ax.set_xlabel('Step (s → s+1)', fontsize=12)
        ax.set_ylabel('Fraction of cells changed', fontsize=12)
        ax.set_title('Hamming Distance per Step (mean ± std)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'hamming_mean_std.png', dpi=160)
        plt.close()
    
    # 2. KL divergence plot (if logits available)
    if 'kl' in results and results['kl']['mean_kl_per_step']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        kl_values = results['kl']['mean_kl_per_step']
        steps = list(range(1, len(kl_values) + 1))
        
        ax.plot(steps, kl_values, 'r-', linewidth=2)
        ax.set_xlabel('Step (s → s+1)', fontsize=12)
        ax.set_ylabel('KL(p^s || p^{s+1})', fontsize=12)
        ax.set_title('Stepwise KL Divergence (mean over puzzles & cells)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'kl_mean.png', dpi=160)
        plt.close()
    
    # 3. Entropy and confidence trajectory
    refine = results['refinement']
    if refine['entropy_trajectory']:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        steps = list(range(len(refine['entropy_trajectory'])))
        
        ax1.plot(steps, refine['entropy_trajectory'], 'b-', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Entropy')
        ax1.set_title('Entropy Over Time (lower = more confident)')
        ax1.grid(alpha=0.3)
        
        ax2.plot(steps, refine['confidence_trajectory'], 'g-', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Max Probability')
        ax2.set_title('Average Confidence Over Time')
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'entropy_confidence_trajectory.png', dpi=160)
        plt.close()
    
    # 4. Confidence distribution at fill
    cert = results['certainty']
    if cert['confidence_distribution']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = list(cert['confidence_distribution'].keys())
        counts = list(cert['confidence_distribution'].values())
        
        ax.bar(bins, counts, color='steelblue', alpha=0.7)
        ax.set_xlabel('Confidence Range')
        ax.set_ylabel('Number of Fills')
        ax.set_title('Confidence Distribution When Filling Cells')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fill_confidence_distribution.png', dpi=160)
        plt.close()
    
    # 5. Algorithm classification scores
    classification = results['classification']
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = list(classification['confidence_scores'].keys())
    scores = [classification['confidence_scores'][s] * 100 for s in strategies]
    
    colors = ['red' if s == classification['primary_strategy'] else 'steelblue' 
              for s in strategies]
    
    ax.barh(strategies, scores, color=colors, alpha=0.7)
    ax.set_xlabel('Confidence Score (%)')
    ax.set_title(f'Algorithm Classification: {classification["primary_strategy"]}')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'algorithm_classification.png', dpi=160)
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze Sudoku solving algorithm')
    parser.add_argument('--intermediate', required=True, 
                       help='Path to intermediate results pickle file')
    parser.add_argument('--inputs', required=True,
                       help='Path to inputs npy file (for givens mask)')
    parser.add_argument('--output', default='results/algorithm_analysis/',
                       help='Output directory for results')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SUDOKU SOLVING ALGORITHM ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    with open(args.intermediate, 'rb') as f:
        interm = pickle.load(f)
    
    inputs = np.load(args.inputs)
    
    # Extract steps
    print(f"  Found {len(interm)} intermediate steps")
    print(f"  Batch size: {interm[0]['predictions'].shape[0]}")
    
    steps_ids = np.stack([step['predictions'].numpy() if hasattr(step['predictions'], 'numpy') 
                          else step['predictions'] for step in interm], axis=0)
    
    # Check if logits available
    if 'logits' in interm[0] and interm[0]['logits'] is not None:
        steps_logits = np.stack([step['logits'].numpy() if hasattr(step['logits'], 'numpy')
                                else step['logits'] for step in interm], axis=0)
        print("  Logits available: Yes")
    else:
        print("  Logits available: No (some analyses will be skipped)")
        steps_logits = None
    
    # Convert to characters
    steps_chars = to_chars(steps_ids)
    givens_chars = to_chars(inputs[:steps_chars.shape[1]])
    givens_mask = (givens_chars != '.')
    
    print(f"  Shape: {steps_chars.shape} (steps, puzzles, cells)")
    
    # Run all analyses
    print("\n" + "="*80)
    print("RUNNING ANALYSES")
    print("="*80)
    
    results = {}
    
    print("\n1. Backtracking Detection...")
    results['backtracking'] = detect_backtracking(steps_chars, givens_mask)
    
    print("2. Monotonicity Analysis...")
    results['monotonic'] = analyze_monotonicity(steps_chars, givens_mask)
    
    if steps_logits is not None:
        print("3. Iterative Refinement Pattern...")
        results['refinement'] = analyze_refinement_pattern(steps_logits, steps_chars, givens_mask)
    else:
        print("3. Iterative Refinement Pattern... SKIPPED (no logits)")
        results['refinement'] = {
            'entropy_decrease_ratio': 0,
            'entropy_start': 0,
            'entropy_end': 0,
            'entropy_reduction': 0,
            'confidence_start': 0,
            'confidence_end': 0,
            'mean_changes_per_step': 0,
            'std_changes_per_step': 0,
            'violation_fix_rate': compute_violation_targeting(steps_chars, givens_mask),
            'entropy_trajectory': [],
            'confidence_trajectory': []
        }
    
    print("\n4. Constraint Propagation Analysis...")
    results['propagation'] = analyze_propagation_waves(steps_chars, givens_mask)
    
    print("5. Hamming Distance Analysis...")
    results['hamming'] = compute_hamming_distance(steps_chars, givens_mask)
    
    if steps_logits is not None:
        print("6. Certainty at Fill Analysis...")
        results['certainty'] = analyze_certainty_at_fill(steps_logits, steps_chars, givens_mask)
        
        print("7. Stepwise KL Divergence...")
        results['kl'] = compute_stepwise_kl(steps_logits, givens_mask)
    else:
        print("6. Certainty at Fill Analysis... SKIPPED (no logits)")
        results['certainty'] = {
            'mean_fill_confidence': 0,
            'median_fill_confidence': 0,
            'min_fill_confidence': 0,
            'max_fill_confidence': 0,
            'low_confidence_fills': 0,
            'high_confidence_fills': 0,
            'confidence_distribution': {}
        }
        print("7. Stepwise KL Divergence... SKIPPED (no logits)")
        results['kl'] = {
            'mean_kl_per_step': [],
            'overall_mean_kl': 0
        }
    
    print("\n8. Algorithm Classification...")
    results['classification'] = classify_algorithm(results)
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print("\n📊 BACKTRACKING METRICS:")
    bt = results['backtracking']
    print(f"  Rollback rate: {bt['rollback_rate']:.1%} ({bt['rollback_steps']} steps)")
    print(f"  Cells with value flips: {bt['cells_with_value_flips']}")
    print(f"  Max changes per cell: {bt['max_changes_per_cell']}")
    
    print("\n📊 MONOTONICITY METRICS:")
    mono = results['monotonic']
    print(f"  Is monotonic: {mono['is_monotonic']}")
    print(f"  Violations: {mono['monotonic_violations']}")
    print(f"  Mean fills per step: {mono['mean_new_fills_per_step']:.2f} ± {mono['std_new_fills_per_step']:.2f}")
    print(f"  Total erases: {mono['total_erases']}")
    
    if steps_logits is not None:
        print("\n📊 REFINEMENT METRICS:")
        refine = results['refinement']
        print(f"  Entropy decrease ratio: {refine['entropy_decrease_ratio']:.1%}")
        print(f"  Entropy: {refine['entropy_start']:.3f} → {refine['entropy_end']:.3f} (Δ={refine['entropy_reduction']:.3f})")
        print(f"  Confidence: {refine['confidence_start']:.1%} → {refine['confidence_end']:.1%}")
        print(f"  Violation fix rate: {refine['violation_fix_rate']:.1%}")
    
    print("\n📊 PROPAGATION METRICS:")
    prop = results['propagation']
    print(f"  Cascade steps: {prop['cascade_steps']}")
    print(f"  Single fills: {prop['single_fills']}")
    print(f"  Cascade ratio: {prop['cascade_ratio']:.1%}")
    
    print("\n📊 HAMMING DISTANCE METRICS:")
    ham = results['hamming']
    print(f"  Overall mean Hamming Δ: {ham['overall_mean_hamming']:.4f}")
    print(f"  Overall std Hamming Δ: {ham['overall_std_hamming']:.4f}")
    print(f"  Number of step transitions: {len(ham['mean_hamming_per_step'])}")
    
    if steps_logits is not None:
        print("\n📊 KL DIVERGENCE METRICS:")
        kl = results['kl']
        print(f"  Overall mean KL: {kl['overall_mean_kl']:.6f}")
        print(f"  Number of step transitions: {len(kl['mean_kl_per_step'])}")
        
        print("\n📊 CERTAINTY METRICS:")
        cert = results['certainty']
        print(f"  Mean fill confidence: {cert['mean_fill_confidence']:.1%}")
        print(f"  High confidence fills (>0.9): {cert['high_confidence_fills']}")
        print(f"  Low confidence fills (<0.6): {cert['low_confidence_fills']}")
    
    print("\n" + "="*80)
    print("🎯 ALGORITHM CLASSIFICATION")
    print("="*80)
    classification = results['classification']
    print(f"\nPrimary Strategy: {classification['primary_strategy']}")
    if classification['is_hybrid']:
        print(f"Secondary Strategy: {classification['secondary_strategy']}")
        print("(Hybrid approach detected)")
    
    print("\nConfidence Scores:")
    for strategy, score in sorted(classification['confidence_scores'].items(), 
                                  key=lambda x: x[1], reverse=True):
        bar = '█' * int(score * 50)
        print(f"  {strategy:25s} {score:5.1%} {bar}")
    
    # Save results
    print(f"\n💾 Saving results to {output_dir}...")
    
    # Save JSON
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save human-readable report
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("SUDOKU SOLVING ALGORITHM ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Primary Strategy: {classification['primary_strategy']}\n")
        if classification['is_hybrid']:
            f.write(f"Secondary Strategy: {classification['secondary_strategy']}\n")
            f.write("Note: Hybrid approach detected\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("-"*80 + "\n\n")
        
        for section, metrics in results.items():
            if section == 'classification':
                continue
            f.write(f"\n{section.upper()}:\n")
            for key, value in metrics.items():
                if not isinstance(value, (list, dict)):
                    f.write(f"  {key}: {value}\n")
    
    # Create visualizations
    print("📈 Creating visualizations...")
    create_visualizations(results, output_dir)
    
    # Save CSV files for Hamming and KL
    if ham['mean_hamming_per_step']:
        with open(output_dir / 'hamming.csv', 'w') as f:
            f.write("step_index,mean_hamming,std_hamming\n")
            for i, (m, s) in enumerate(zip(ham['mean_hamming_per_step'], ham['std_hamming_per_step']), start=1):
                f.write(f"{i},{m},{s}\n")
    
    if steps_logits is not None and results['kl']['mean_kl_per_step']:
        with open(output_dir / 'kl.csv', 'w') as f:
            f.write("step_index,mean_kl\n")
            for i, k in enumerate(results['kl']['mean_kl_per_step'], start=1):
                f.write(f"{i},{k}\n")
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir.resolve()}")
    
    return results

if __name__ == "__main__":
    main()
