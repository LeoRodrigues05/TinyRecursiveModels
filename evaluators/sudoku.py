from typing import Dict, Sequence, Optional, Any
import os
import json

import torch
import numpy as np
import torch.distributed as dist

from dataset.common import PuzzleDatasetMetadata

class SudokuEvaluator:
    required_outputs = {"preds", "puzzle_identifiers"}

    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.eval_metadata = eval_metadata
        # Local buffer of predictions (in order seen)
        self._local_preds = []

    def begin_eval(self):
        self._local_preds = []

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # preds expected to contain key 'preds' with shape (B, seq_len)
        if "preds" in preds:
            p = preds["preds"].cpu().numpy()
            # Store as plain numpy arrays
            for row in p:
                self._local_preds.append(row.tolist())
        else:
            # Nothing to collect
            return

    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[dist.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        # If distributed world size is 1 or the default process group isn't initialized,
        # skip distributed collectives and use local predictions directly.
        if world_size == 1 or (not dist.is_available()) or (not dist.is_initialized()):
            if rank != 0:
                return None

            all_preds = list(self._local_preds)
        else:
            # Gather local predictions to rank 0
            gathered = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(self._local_preds, gathered, dst=0, group=group)
            if rank != 0:
                return None

            # Concatenate gathered lists
            all_preds = []
            for g in gathered:
                if g:
                    all_preds.extend(g)

        all_preds = np.array(all_preds, dtype=np.int64)

        # Save as numpy and a json mapping
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, "submission_preds.npy"), all_preds)

            # Also save a simple JSON mapping of index -> flattened list
            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump({str(i): pred.tolist() for i, pred in enumerate(all_preds)}, f)

        # No scalar metrics computed here
        return None
