# Guide: Capturing Intermediate Sudoku Solving Results from TRM Model

## Overview

The TinyRecursiveReasoningModel (TRM) solves Sudoku puzzles through an **Adaptive Computation Time (ACT)** mechanism that iteratively refines its solution. This guide explains the model's architecture and identifies exactly where you need to make code changes to capture intermediate solving steps for a single puzzle.

---

## Architecture Overview

### The Recursive Solving Loop (Model-Level)

**File:** `models/recursive_reasoning/trm.py`

The key insight is that TRM does **NOT** solve puzzles in a single forward pass. Instead, it operates in a **recursive loop** at the evaluation level:

```
Evaluation Loop (pretrain.py, line ~408):
    for each batch:
        carry = model.initial_carry(batch)  # Initialize carry for batch
        while True:
            carry, loss, metrics, preds, all_finish = model(carry=carry, batch=batch, ...)
            # ^^ This is ONE STEP of recursive reasoning
            if all_finish:
                break
```

**Each call to `model(carry=carry, batch=batch, ...)` is ONE iteration step where the model:**
1. Takes the current state (carry)
2. Processes it through transformer layers (hierarchical + local reasoning)
3. Produces intermediate predictions
4. Decides whether each puzzle is solved (ACT halting decision)
5. Returns updated state for next iteration

### Carry Object (State Container)

**File:** `models/recursive_reasoning/trm.py`, lines 29-43 (dataclass definitions)

```python
@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor  # Hierarchical hidden state (B x SeqLen x H)
    z_L: torch.Tensor  # Local hidden state       (B x SeqLen x H)

@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor        # Current step number per example (B,)
    halted: torch.Tensor       # Whether each example is solved (B,)
    current_data: Dict[str, Tensor]  # Current batch data
```

**Key insight:** `steps` tracks which iteration each puzzle is on, and `halted` indicates which puzzles are finished.

### Model Forward Pass (Single Step)

**File:** `models/recursive_reasoning/trm.py`, lines 164-210

```python
def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]):
    # Step 1: Reset state for newly halted sequences
    new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
    new_steps = torch.where(carry.halted, 0, carry.steps)
    
    # Step 2: Update current data (use fresh batch for halted examples)
    new_current_data = {...}
    
    # Step 3: Run transformer forward pass (THIS IS WHERE SOLVING HAPPENS)
    new_inner_carry, logits, (q_halt_logits, q_continue_logits) = 
        self.inner(new_inner_carry, new_current_data)
    
    # Step 4: Output predictions and halting decisions
    outputs = {
        "logits": logits,                      # Raw predictions (B x SeqLen x 10)
        "q_halt_logits": q_halt_logits,        # Should we halt? (B,)
        "q_continue_logits": q_continue_logits # Continue confidence (B,)
    }
    
    # Step 5: Determine which puzzles are solved
    halted = is_last_step | (q_halt_logits > q_continue_logits)
    
    return TinyRecursiveReasoningModel_ACTV1Carry(...), outputs
```

### Inner Forward Pass (Transformer Layers)

**File:** `models/recursive_reasoning/trm.py`, lines 176-210 (in `TinyRecursiveReasoningModel_ACTV1_Inner`)

```python
def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict):
    # 1. Embed input tokens
    input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    
    # 2. Hierarchical-Local recursive computation (multiple H_cycles × L_cycles)
    z_H, z_L = carry.z_H, carry.z_L
    
    # First H_cycles-1 without gradient (optimization)
    with torch.no_grad():
        for _H_step in range(self.config.H_cycles - 1):
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)
    
    # Last cycle WITH gradient
    for _L_step in range(self.config.L_cycles):
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
    z_H = self.L_level(z_H, z_L, **seq_info)
    
    # 3. Produce outputs
    output = self.lm_head(z_H)[:, self.puzzle_emb_len:]  # (B x 81 x 10) predictions
    q_logits = self.q_head(z_H[:, 0])                    # (B x 2) halt decision
    
    return new_carry, output, (q_logits[..., 0], q_logits[..., 1])
```

---

## Where to Capture Intermediate Results

### **OPTION 1: Capture at Evaluation Loop Level (RECOMMENDED - Simplest)**

**File:** `pretrain.py`, lines 405-445

**Current code:**
```python
while True:
    carry, loss, metrics, preds, all_finish = train_state.model(
        carry=carry, batch=batch, return_keys=return_keys
    )
    inference_steps += 1
    if all_finish:
        break
```

**What to modify:**
Add a data structure to collect intermediate predictions per step:

```python
# NEW: Before the while loop (line ~408)
intermediate_results = []  # List of dicts, one per step

while True:
    # Capture BEFORE the step
    step_num = inference_steps
    active_mask = ~carry.halted  # Which examples are still solving
    
    # Run one step
    carry, loss, metrics, preds, all_finish = train_state.model(
        carry=carry, batch=batch, return_keys=return_keys
    )
    inference_steps += 1
    
    # Capture AFTER the step
    step_data = {
        "step": step_num,
        "predictions": preds["preds"].cpu(),  # (B x 81) current predictions
        "logits": preds.get("logits", None).cpu() if "logits" in preds else None,  # (B x 81 x 10)
        "q_halt_logits": preds.get("q_halt_logits", None).cpu() if "q_halt_logits" in preds else None,  # (B,)
        "steps_per_example": carry.steps.cpu(),  # (B,) - which step each example is on
        "halted": carry.halted.cpu(),  # (B,) - which examples finished this step
        "active_mask": active_mask.cpu(),  # (B,) - which were solving during this step
    }
    intermediate_results.append(step_data)
    
    if all_finish:
        break

# NEW: After loop - save intermediate results
if rank == 0:
    import pickle
    output_dir = os.path.dirname(config.checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save per-batch intermediate results
    with open(os.path.join(output_dir, "intermediate_results.pkl"), "wb") as f:
        pickle.dump(intermediate_results, f)
```

**Advantages:**
- ✅ Simple - minimal code changes
- ✅ Captures entire batch per step
- ✅ Records the halting decisions
- ✅ Easy to analyze step-by-step evolution

**Output format:**
```
intermediate_results = [
    {
        "step": 0,
        "predictions": torch.Size([batch_size, 81]),  # First predictions
        "logits": torch.Size([batch_size, 81, 10]),   # Raw logits for digits 0-9
        "q_halt_logits": torch.Size([batch_size]),    # Halt confidence
        "steps_per_example": torch.Size([batch_size]), # 1,1,1,...
        "halted": torch.Size([batch_size]),           # Which finished at step 0
        "active_mask": torch.Size([batch_size]),      # All True at step 0
    },
    {
        "step": 1,
        "predictions": torch.Size([batch_size, 81]),  # Updated predictions
        ...
    },
    ...
]
```

---

### **OPTION 2: Capture at Model Level (Finer Granularity)**

**File:** `models/recursive_reasoning/trm.py`

Modify the model to collect intermediate states during the transformer passes:

```python
# In TinyRecursiveReasoningModel_ACTV1_Inner.forward() (line ~188)

def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict):
    input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    
    z_H, z_L = carry.z_H, carry.z_L
    intermediate_hidden_states = []  # NEW: Track internal states
    
    # H_cycles-1 without grad
    with torch.no_grad():
        for _H_step in range(self.config.H_cycles - 1):
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)
            intermediate_hidden_states.append({  # NEW
                "H_cycle": _H_step,
                "z_H": z_H.detach().clone(),
                "z_L": z_L.detach().clone(),
            })
    
    # Last cycle with grad
    for _L_step in range(self.config.L_cycles):
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
    z_H = self.L_level(z_H, z_L, **seq_info)
    intermediate_hidden_states.append({  # NEW
        "H_cycle": self.config.H_cycles - 1,
        "z_H": z_H.clone(),
        "z_L": z_L.clone(),
    })
    
    # Output predictions
    output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
    q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
    
    # NEW: Return intermediate states in outputs dict
    return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), intermediate_hidden_states
```

**Advantages:**
- ✅ Can track hidden state evolution through transformer layers
- ✅ Understand which layers contribute to solving
- ✅ See z_H (hierarchical) vs z_L (local) evolution

**Disadvantages:**
- ❌ More code changes needed (affects multiple layers)
- ❌ Larger memory footprint (storing all intermediate tensors)

---

### **OPTION 3: Hybrid - Per-Puzzle Extraction (Best for Analysis)**

Create a dedicated function to extract intermediate results for a **single puzzle** from a batch:

**File:** Create new file `scripts/extract_single_puzzle_trajectory.py`

```python
import torch
import pickle
import numpy as np
from pathlib import Path

def extract_single_puzzle_trajectory(
    model,
    batch: Dict[str, torch.Tensor],
    puzzle_idx: int,  # Index within batch
    config=None
):
    """
    Extract the solving trajectory for a single puzzle.
    Returns step-by-step predictions as the model refines its solution.
    """
    
    # Extract single puzzle from batch
    single_puzzle = {k: v[puzzle_idx:puzzle_idx+1] for k, v in batch.items()}
    
    trajectory = {
        "puzzle_idx": puzzle_idx,
        "input": single_puzzle["inputs"].cpu().numpy(),
        "label": single_puzzle["labels"].cpu().numpy(),
        "steps": [],
    }
    
    # Run model inference with capture
    carry = model.initial_carry(single_puzzle)
    step_num = 0
    
    while True:
        carry, loss, metrics, preds, all_finish = model(
            carry=carry, batch=single_puzzle, return_keys={"preds", "logits"}
        )
        
        # Extract predictions for this puzzle
        pred_digits = preds["preds"][0].cpu().numpy()  # (81,)
        logits = preds.get("logits", None)
        confidence = torch.max(torch.softmax(logits[0], dim=-1), dim=-1)[0].cpu().numpy() if logits is not None else None
        
        trajectory["steps"].append({
            "step_number": step_num,
            "predictions": pred_digits,
            "confidence": confidence,
            "is_halted": carry.halted[0].item(),
            "q_halt_logits": preds["q_halt_logits"][0].item() if "q_halt_logits" in preds else None,
        })
        
        step_num += 1
        if all_finish:
            break
    
    return trajectory


# Usage in evaluate():
if rank == 0:
    # Extract a few example puzzle trajectories
    example_indices = [0, batch["inputs"].shape[0] // 2, -1]  # First, middle, last
    trajectories = []
    
    for idx in example_indices:
        traj = extract_single_puzzle_trajectory(
            train_state.model, batch, idx
        )
        trajectories.append(traj)
    
    # Save trajectories
    output_dir = os.path.dirname(config.checkpoint_path)
    with open(os.path.join(output_dir, "puzzle_trajectories.pkl"), "wb") as f:
        pickle.dump(trajectories, f)
```

**Output format (single puzzle):**
```python
trajectory = {
    "puzzle_idx": 0,
    "input": np.ndarray(81,),         # Input Sudoku as flat tokens
    "label": np.ndarray(81,),         # Ground truth solution
    "steps": [
        {
            "step_number": 0,
            "predictions": np.ndarray(81,),  # Digit predictions (0-9)
            "confidence": np.ndarray(81,),   # Confidence per cell (0-1)
            "is_halted": False,
            "q_halt_logits": -2.34,
        },
        {
            "step_number": 1,
            "predictions": np.ndarray(81,),  # Refined predictions
            "confidence": np.ndarray(81,),
            "is_halted": False,
            "q_halt_logits": -0.5,
        },
        {
            "step_number": 2,
            "predictions": np.ndarray(81,),  # Final predictions
            "confidence": np.ndarray(81,),
            "is_halted": True,               # Model decided to halt
            "q_halt_logits": 3.2,
        },
    ]
}
```

**Advantages:**
- ✅ Focused on single puzzles (easier analysis)
- ✅ Compact output (only one puzzle per file)
- ✅ Easy to visualize step-by-step grid evolution
- ✅ Can compute "convergence" metrics

---

## Step-by-Step Implementation Guide

### Quick Start (5 minutes)

1. **Add intermediate capture to `pretrain.py` (OPTION 1):**
   - File: `/home/ubuntu/TinyRecursiveModels/pretrain.py`
   - Location: Line 408 (in the `while True:` evaluation loop)
   - Insert the code from OPTION 1 above

2. **Add to required_outputs:**
   - In config, ensure `eval_save_outputs` includes what you need:
   ```yaml
   eval_save_outputs: ["preds", "logits", "q_halt_logits"]
   ```

3. **Run evaluation:**
   ```bash
   cd /home/ubuntu/TinyRecursiveModels
   python pretrain.py \
     --config-path=checkpoints \
     --config-name=all_config_eval_25pct \
     load_checkpoint=checkpoints/step_6510.pt \
     epochs=1
   ```

4. **Load results:**
   ```python
   import pickle
   with open("checkpoints/intermediate_results.pkl", "rb") as f:
       results = pickle.load(f)
   
   # results[i] = data for step i
   # Each contains predictions for all examples in batch
   ```

---

## File Locations Summary

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Evaluation Loop | `pretrain.py` | 405-445 | **👈 Add intermediate capture HERE** |
| Carry Definition | `trm.py` | 29-43 | State container (z_H, z_L, steps, halted) |
| Model Forward | `trm.py` | 164-210 | Single step of solving |
| Inner Forward | `trm.py` | 176-210 | Transformer passes (hierarchical-local) |
| Loss & Metrics | `models/losses.py` | 30-60 | How predictions are extracted (`argmax` on logits) |
| Report Generation | `scripts/generate_report_from_preds.py` | - | Convert predictions to CSV/HTML |

---

## Key Variables to Monitor

When capturing intermediate results, these variables are most useful:

```python
carry.steps              # Which iteration each example is on
carry.halted             # Which examples have finished solving
carry.inner_carry.z_H    # Hierarchical hidden state (abstract reasoning)
carry.inner_carry.z_L    # Local hidden state (cell-level reasoning)
preds["preds"]           # Current Sudoku digit predictions (81,)
preds["logits"]          # Raw logits before argmax (81 x 10)
preds["q_halt_logits"]   # Halt decision confidence
```

---

## Example: Visualizing Single Puzzle Evolution

```python
import numpy as np
import matplotlib.pyplot as plt

# After extracting trajectory
def visualize_trajectory(trajectory):
    steps_data = trajectory["steps"]
    
    fig, axes = plt.subplots(1, len(steps_data), figsize=(4*len(steps_data), 4))
    
    for step_idx, step_data in enumerate(steps_data):
        ax = axes[step_idx] if len(steps_data) > 1 else axes
        
        # Reshape to 9x9 grid
        grid = step_data["predictions"].reshape(9, 9)
        
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=9)
        ax.set_title(f"Step {step_data['step_number']}\nHalt: {step_data['is_halted']}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add text annotations
        for i in range(9):
            for j in range(9):
                ax.text(j, i, str(grid[i, j]), ha="center", va="center")
    
    plt.tight_layout()
    plt.savefig("puzzle_evolution.png")
    plt.show()

# Usage
visualize_trajectory(trajectories[0])
```

---

## Questions to Answer

**Q: How many steps does a typical puzzle take?**
- A: Check the `.pkl` output - count the steps list length. Likely 3-10 steps depending on puzzle difficulty.

**Q: Which cells does the model refine most?**
- A: Compare predictions across steps - cells that change frequently are harder to solve.

**Q: Is there a correlation between confidence and accuracy?**
- A: Plot step-number vs cell-confidence for correct vs incorrect cells.

**Q: Does the model halt when it should?**
- A: Compare `is_halted` with whether `predictions == labels`.
