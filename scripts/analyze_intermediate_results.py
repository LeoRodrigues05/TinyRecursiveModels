"""
Analyze and visualize intermediate solving steps captured during model evaluation.

Usage:
    python3 scripts/analyze_intermediate_results.py \\
      --intermediate-dir checkpoints/ \\
      --output-dir analysis_output/

This will load all intermediate_results_*.pkl files and create visualizations.
"""

import pickle
import numpy as np
import argparse
import json
from pathlib import Path
import os


def load_intermediate_results(intermediate_dir):
    """Load all intermediate results files from a directory."""
    intermediate_dir = Path(intermediate_dir)
    pkl_files = sorted(intermediate_dir.glob("intermediate_results_*.pkl"))
    
    if not pkl_files:
        print(f"No intermediate_results_*.pkl files found in {intermediate_dir}")
        return None
    
    all_results = []
    for pkl_file in pkl_files:
        print(f"Loading {pkl_file.name}...")
        with open(pkl_file, "rb") as f:
            batch_results = pickle.load(f)
            all_results.append({
                "file": pkl_file.name,
                "results": batch_results
            })
    
    return all_results


def analyze_single_puzzle_trajectory(intermediate_results, puzzle_idx=0, batch_idx=0):
    """Extract the solving trajectory for a single puzzle."""
    if intermediate_results is None or len(intermediate_results) == 0:
        print("No intermediate results found")
        return None
    
    if batch_idx >= len(intermediate_results):
        print(f"Batch index {batch_idx} out of range (max {len(intermediate_results)-1})")
        batch_idx = 0
    
    batch_data = intermediate_results[batch_idx]["results"]
    
    trajectory = {
        "puzzle_idx": puzzle_idx,
        "batch_idx": batch_idx,
        "file": intermediate_results[batch_idx]["file"],
        "steps": [],
    }
    
    for step_data in batch_data:
        step_info = {
            "step_number": step_data["step"],
            "predictions": step_data["predictions"][puzzle_idx].numpy() if step_data["predictions"] is not None else None,
            "logits": step_data["logits"][puzzle_idx].numpy() if step_data["logits"] is not None else None,
            "q_halt_logits": step_data["q_halt_logits"][puzzle_idx].item() if step_data["q_halt_logits"] is not None else None,
            "q_continue_logits": step_data["q_continue_logits"][puzzle_idx].item() if step_data["q_continue_logits"] is not None else None,
            "is_halted": step_data["halted"][puzzle_idx].item(),
            "was_active": step_data["active_mask"][puzzle_idx].item(),
        }
        
        # Compute confidence (max softmax across digits)
        if step_info["logits"] is not None:
            softmax = np.exp(step_info["logits"]) / np.sum(np.exp(step_info["logits"]), axis=-1, keepdims=True)
            step_info["max_confidence"] = np.max(softmax, axis=-1)  # Per-cell max confidence
            step_info["mean_confidence"] = np.mean(step_info["max_confidence"])
        
        trajectory["steps"].append(step_info)
    
    return trajectory


def print_puzzle_trajectory(trajectory):
    """Print a summary of a puzzle's solving trajectory."""
    if trajectory is None:
        return
    
    print(f"\n{'='*80}")
    print(f"Puzzle {trajectory['puzzle_idx']} from batch {trajectory['batch_idx']}")
    print(f"File: {trajectory['file']}")
    print(f"{'='*80}")
    print(f"{'Step':<6} {'Halted':<8} {'Was Active':<12} {'Q_Halt':<10} {'Mean Conf':<12}")
    print(f"{'-'*80}")
    
    for step in trajectory["steps"]:
        halted_str = "✓" if step["is_halted"] else "-"
        active_str = "✓" if step["was_active"] else "-"
        q_halt = f"{step['q_halt_logits']:.3f}" if step["q_halt_logits"] is not None else "N/A"
        mean_conf = f"{step['mean_confidence']:.3f}" if "mean_confidence" in step else "N/A"
        
        print(f"{step['step_number']:<6} {halted_str:<8} {active_str:<12} {q_halt:<10} {mean_conf:<12}")


def create_convergence_analysis(intermediate_results):
    """Analyze convergence statistics across all puzzles and batches."""
    if intermediate_results is None or len(intermediate_results) == 0:
        return None
    
    total_puzzles = 0
    total_steps = 0
    puzzles_halting_per_step = {}
    
    for batch in intermediate_results:
        batch_results = batch["results"]
        
        if len(batch_results) == 0:
            continue
        
        batch_size = batch_results[0]["halted"].shape[0]
        total_puzzles += batch_size
        
        for step_idx, step_data in enumerate(batch_results):
            halted_this_step = step_data["halted"].sum().item()
            active = (~step_data["halted"]).sum().item()
            
            if step_idx not in puzzles_halting_per_step:
                puzzles_halting_per_step[step_idx] = {
                    "halted": 0,
                    "active": 0,
                    "batches": 0,
                }
            
            puzzles_halting_per_step[step_idx]["halted"] += halted_this_step
            puzzles_halting_per_step[step_idx]["active"] += active
            puzzles_halting_per_step[step_idx]["batches"] += 1
            
            total_steps = max(total_steps, step_idx + 1)
    
    print(f"\n{'='*80}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total puzzles: {total_puzzles}")
    print(f"Total steps: {total_steps}")
    print(f"\n{'Step':<6} {'Active':<12} {'Halted':<12} {'Halt %':<12}")
    print(f"{'-'*80}")
    
    cumulative_halted = 0
    for step_idx in sorted(puzzles_halting_per_step.keys()):
        data = puzzles_halting_per_step[step_idx]
        active_pct = (data["active"] / total_puzzles) * 100
        cumulative_halted += data["halted"]
        halt_pct = (cumulative_halted / total_puzzles) * 100
        
        print(f"{step_idx:<6} {active_pct:<12.1f}% {data['halted']:<12} {halt_pct:<12.1f}%")


def save_trajectory_json(trajectory, output_path):
    """Save a trajectory as JSON for easier inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_trajectory = {
        "puzzle_idx": trajectory["puzzle_idx"],
        "batch_idx": trajectory["batch_idx"],
        "file": trajectory["file"],
        "steps": []
    }
    
    for step in trajectory["steps"]:
        json_step = {
            "step_number": int(step["step_number"]),
            "q_halt_logits": float(step["q_halt_logits"]) if step["q_halt_logits"] is not None else None,
            "q_continue_logits": float(step["q_continue_logits"]) if step["q_continue_logits"] is not None else None,
            "is_halted": bool(step["is_halted"]),
            "was_active": bool(step["was_active"]),
        }
        
        if "mean_confidence" in step:
            json_step["mean_confidence"] = float(step["mean_confidence"])
        
        # Include first few predictions for inspection
        if step["predictions"] is not None:
            json_step["predictions_first_10"] = step["predictions"][:10].tolist()
        
        json_trajectory["steps"].append(json_step)
    
    with open(output_path, "w") as f:
        json.dump(json_trajectory, f, indent=2)
    
    print(f"Saved trajectory to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze intermediate solving results")
    parser.add_argument(
        "--intermediate-dir",
        type=str,
        default="checkpoints",
        help="Directory containing intermediate_results_*.pkl files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Directory to save analysis outputs"
    )
    parser.add_argument(
        "--puzzle-idx",
        type=int,
        default=0,
        help="Puzzle index within batch to analyze in detail"
    )
    parser.add_argument(
        "--batch-idx",
        type=int,
        default=0,
        help="Batch index to analyze in detail"
    )
    
    args = parser.parse_args()
    
    # Load all intermediate results
    intermediate_results = load_intermediate_results(args.intermediate_dir)
    
    if intermediate_results is None:
        return
    
    print(f"\nLoaded {len(intermediate_results)} batch(es) of intermediate results")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze convergence
    create_convergence_analysis(intermediate_results)
    
    # Analyze single puzzle trajectory
    trajectory = analyze_single_puzzle_trajectory(
        intermediate_results,
        puzzle_idx=args.puzzle_idx,
        batch_idx=args.batch_idx
    )
    
    if trajectory is not None:
        print_puzzle_trajectory(trajectory)
        
        # Save as JSON
        json_output = os.path.join(
            args.output_dir,
            f"trajectory_batch{args.batch_idx}_puzzle{args.puzzle_idx}.json"
        )
        save_trajectory_json(trajectory, json_output)
    
    print(f"\n✓ Analysis complete. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
