#!/usr/bin/env python3
"""
Example: Analyzing Intermediate Sudoku Solving Results

This script demonstrates how to load and analyze intermediate results captured
during model evaluation to understand how the TRM model solves Sudoku puzzles
iteratively.

Run with: python3 examples/analyze_sudoku_solving.py
"""

import pickle
import numpy as np
from pathlib import Path
import json


def example_1_basic_loading():
    """Example 1: Load intermediate results and examine structure."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Loading Intermediate Results")
    print("="*80)
    
    pkl_file = "checkpoints/intermediate_results_step_0_batch_1.pkl"
    
    try:
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)
        
        print(f"✓ Loaded {pkl_file}")
        print(f"  Number of steps: {len(results)}")
        print(f"  Batch size: {results[0]['predictions'].shape[0]}")
        print(f"  Cells per puzzle: {results[0]['predictions'].shape[1]}")
        
        # Show structure of first step
        step_0 = results[0]
        print(f"\nFirst step contains:")
        for key, value in step_0.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value).__name__}")
    
    except FileNotFoundError:
        print(f"⚠ File not found: {pkl_file}")
        print("  Run evaluation first to generate intermediate results")


def example_2_track_single_puzzle():
    """Example 2: Track a single puzzle through all solving steps."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Tracking Single Puzzle Evolution")
    print("="*80)
    
    pkl_file = "checkpoints/intermediate_results_step_0_batch_1.pkl"
    
    try:
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)
        
        puzzle_idx = 0  # Track first puzzle in batch
        
        print(f"\nTracking puzzle {puzzle_idx}:")
        print(f"{'Step':<6} {'Pred (1st 10)':<40} {'Halted':<8} {'Q_Halt':<10}")
        print("-" * 70)
        
        for step_data in results:
            preds = step_data["predictions"][puzzle_idx][:10]
            halted = step_data["halted"][puzzle_idx].item()
            q_halt = step_data["q_halt_logits"][puzzle_idx].item()
            
            print(f"{step_data['step']:<6} {str(preds.numpy().tolist()):<40} "
                  f"{str(halted):<8} {q_halt:>10.3f}")
    
    except FileNotFoundError:
        print(f"⚠ File not found: {pkl_file}")


def example_3_confidence_evolution():
    """Example 3: Analyze how model confidence evolves."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Confidence Evolution")
    print("="*80)
    
    pkl_file = "checkpoints/intermediate_results_step_0_batch_1.pkl"
    
    try:
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)
        
        puzzle_idx = 0
        
        print(f"\nConfidence stats for puzzle {puzzle_idx}:")
        print(f"{'Step':<6} {'Mean Conf':<15} {'Min Conf':<15} {'Max Conf':<15}")
        print("-" * 55)
        
        for step_data in results:
            logits = step_data["logits"][puzzle_idx].numpy()  # (81, 10)
            
            # Convert to probabilities
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            max_probs = np.max(probs, axis=-1)  # Confidence per cell
            
            mean_conf = np.mean(max_probs)
            min_conf = np.min(max_probs)
            max_conf = np.max(max_probs)
            
            print(f"{step_data['step']:<6} {mean_conf:>14.4f} {min_conf:>14.4f} {max_conf:>14.4f}")
    
    except FileNotFoundError:
        print(f"⚠ File not found: {pkl_file}")


def example_4_halting_analysis():
    """Example 4: Analyze halting behavior."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Halting Behavior Analysis")
    print("="*80)
    
    pkl_file = "checkpoints/intermediate_results_step_0_batch_1.pkl"
    
    try:
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)
        
        batch_size = results[0]["predictions"].shape[0]
        
        print(f"\nBatch size: {batch_size} puzzles")
        print(f"{'Step':<6} {'Still Active':<15} {'Newly Halted':<15} {'Total Halted':<15}")
        print("-" * 55)
        
        total_halted = 0
        for step_data in results:
            halted = step_data["halted"].numpy()
            total_halted = np.sum(halted)
            
            if step_data["step"] > 0:
                prev_halted = results[step_data["step"] - 1]["halted"].numpy()
                newly_halted = np.sum(halted & ~prev_halted)
                still_active = batch_size - total_halted
            else:
                newly_halted = total_halted
                still_active = batch_size - total_halted
            
            print(f"{step_data['step']:<6} {still_active:>14} {newly_halted:>14} {total_halted:>14}")
    
    except FileNotFoundError:
        print(f"⚠ File not found: {pkl_file}")


def example_5_prediction_changes():
    """Example 5: Track prediction changes between steps."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Prediction Changes Between Steps")
    print("="*80)
    
    pkl_file = "checkpoints/intermediate_results_step_0_batch_1.pkl"
    
    try:
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)
        
        puzzle_idx = 0
        
        print(f"\nTracking prediction changes for puzzle {puzzle_idx}:")
        print(f"{'Step':<6} {'Cells Changed':<15} {'% Changed':<15}")
        print("-" * 40)
        
        for i, step_data in enumerate(results):
            if i == 0:
                print(f"{step_data['step']:<6} {'N/A':<15} {'N/A':<15}")
            else:
                prev_preds = results[i-1]["predictions"][puzzle_idx].numpy()
                curr_preds = step_data["predictions"][puzzle_idx].numpy()
                
                changes = np.sum(prev_preds != curr_preds)
                pct_changed = (changes / 81) * 100
                
                print(f"{step_data['step']:<6} {changes:<14} {pct_changed:>13.1f}%")
    
    except FileNotFoundError:
        print(f"⚠ File not found: {pkl_file}")


def example_6_batch_statistics():
    """Example 6: Aggregate statistics across entire batch."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Batch-Wide Statistics")
    print("="*80)
    
    pkl_file = "checkpoints/intermediate_results_step_0_batch_1.pkl"
    
    try:
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)
        
        print(f"\nStatistics across all puzzles:")
        print(f"{'Step':<6} {'Avg Steps':<15} {'% Halted':<15} {'Mean Conf':<15}")
        print("-" * 55)
        
        for step_data in results:
            steps = step_data["steps_per_example"].float().mean().item()
            halted_pct = step_data["halted"].float().mean().item() * 100
            
            # Confidence
            logits = step_data["logits"].numpy()  # (batch_size, 81, 10)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            max_probs = np.max(probs, axis=-1)  # (batch_size, 81)
            mean_conf = np.mean(max_probs)
            
            print(f"{step_data['step']:<6} {steps:>14.2f} {halted_pct:>13.1f}% {mean_conf:>14.4f}")
    
    except FileNotFoundError:
        print(f"⚠ File not found: {pkl_file}")


def example_7_export_trajectory_json():
    """Example 7: Export a puzzle trajectory as JSON."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Exporting Trajectory as JSON")
    print("="*80)
    
    pkl_file = "checkpoints/intermediate_results_step_0_batch_1.pkl"
    output_file = "example_trajectory.json"
    
    try:
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)
        
        puzzle_idx = 0
        trajectory = {
            "puzzle_idx": puzzle_idx,
            "steps": []
        }
        
        for step_data in results:
            step_dict = {
                "step_number": int(step_data["step"]),
                "predictions_first_10": step_data["predictions"][puzzle_idx][:10].numpy().tolist(),
                "is_halted": bool(step_data["halted"][puzzle_idx].item()),
                "q_halt_logits": float(step_data["q_halt_logits"][puzzle_idx].item()),
            }
            trajectory["steps"].append(step_dict)
        
        with open(output_file, "w") as f:
            json.dump(trajectory, f, indent=2)
        
        print(f"✓ Exported trajectory to {output_file}")
        print(f"  Total steps: {len(trajectory['steps'])}")
    
    except FileNotFoundError:
        print(f"⚠ File not found: {pkl_file}")


def main():
    """Run all examples."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  ANALYZING INTERMEDIATE SUDOKU SOLVING RESULTS".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Run examples
    example_1_basic_loading()
    example_2_track_single_puzzle()
    example_3_confidence_evolution()
    example_4_halting_analysis()
    example_5_prediction_changes()
    example_6_batch_statistics()
    example_7_export_trajectory_json()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
1. Modify these examples for your analysis
2. Use analyze_intermediate_results.py for automated analysis
3. See INTERMEDIATE_RESULTS_USAGE.md for more examples
4. Create custom visualizations based on the data

For questions, see:
- INTERMEDIATE_QUICK_REF.md (quick reference)
- INTERMEDIATE_RESULTS_USAGE.md (detailed guide)
- INTERMEDIATE_CAPTURE_SUMMARY.md (implementation details)
""")


if __name__ == "__main__":
    main()
