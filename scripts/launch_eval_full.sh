#!/bin/bash
# Launch evaluation on full test set in background

set -e

# Configuration
CHECKPOINT="checkpoints/step_6510.pt"
OUTPUT_DIR="checkpoints/eval_full"
CONFIG_PATH="checkpoints"
CONFIG_NAME="all_config_eval"
BATCH_SIZE=768
LOG_FILE="eval_full_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="eval_full.pid"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting full test set evaluation...${NC}"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Log file: $LOG_FILE"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Run evaluation in background with nohup
nohup python pretrain.py \
  --config-path "$CONFIG_PATH" \
  --config-name "$CONFIG_NAME" \
  load_checkpoint="$CHECKPOINT" \
  checkpoint_path="$OUTPUT_DIR" \
  epochs=1 \
  eval_interval=1 \
  min_eval_interval=0 \
  global_batch_size=$BATCH_SIZE \
  > "$LOG_FILE" 2>&1 &

# Save process ID
EVAL_PID=$!
echo $EVAL_PID > "$PID_FILE"

echo -e "${GREEN}✓ Evaluation launched!${NC}"
echo "Process ID: $EVAL_PID"
echo ""
echo -e "${YELLOW}Monitoring commands:${NC}"
echo "  View log:        tail -f $LOG_FILE"
echo "  Check progress:  grep -i 'accuracy\\|loss\\|step' $LOG_FILE | tail -20"
echo "  Check if running: ps -p $EVAL_PID"
echo "  Kill process:    kill $EVAL_PID"
echo ""
echo "You can now safely logout. The evaluation will continue running."
