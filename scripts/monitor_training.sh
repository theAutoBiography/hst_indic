#!/bin/bash
# Monitor ByT5 training - GPU, CPU, Memory, and Progress

echo "================================================================"
echo "                  ByT5 TRAINING MONITOR"
echo "================================================================"
echo ""

# Find the training process
PROCESS=$(ps aux | grep "train_byt5" | grep -v grep | grep python)

if [ -z "$PROCESS" ]; then
    echo "❌ No training process found"
    echo ""
    echo "Check if training is running:"
    echo "  tmux ls"
    echo "  tmux attach -t byt5_training"
    echo ""
    exit 1
fi

# Extract PID
PID=$(echo "$PROCESS" | awk '{print $2}')
CPU=$(echo "$PROCESS" | awk '{print $3}')
MEM=$(echo "$PROCESS" | awk '{print $4}')
COMMAND=$(echo "$PROCESS" | awk '{print $11}')

echo "✅ Training process found"
echo ""
echo "Process Info:"
echo "  PID:     $PID"
echo "  Command: $COMMAND"
echo "  CPU:     ${CPU}%"
echo "  Memory:  ${MEM}%"
echo ""

# GPU Info
echo "----------------------------------------------------------------"
echo "GPU Status:"
echo "----------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s\n    Utilization: %s%%\n    Memory: %s / %s MB (%.1f%%)\n    Temperature: %s°C\n\n",
                 $1, $2, $3, $4, $5, ($4/$5)*100, $6}'

# Check if process is using GPU
echo "----------------------------------------------------------------"
echo "GPU Process Info:"
echo "----------------------------------------------------------------"
GPU_PROCESS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader | grep $PID)

if [ -z "$GPU_PROCESS" ]; then
    echo "  ⚠️  Process not found on GPU (may be loading)"
else
    GPU_MEM=$(echo "$GPU_PROCESS" | awk -F', ' '{print $3}')
    echo "  ✅ Process using GPU"
    echo "  GPU Memory: $GPU_MEM"
fi
echo ""

# Training progress (if log file exists)
echo "----------------------------------------------------------------"
echo "Training Progress:"
echo "----------------------------------------------------------------"

# Check in current directory and common locations
LOG_FILES=(
    "training.log"
    "models/byt5-sandhi-small/logs/events.out.tfevents.*"
    "/data/ramanan/byt5/training.log"
)

LOG_FOUND=false
for LOG in "${LOG_FILES[@]}"; do
    if ls $LOG 2>/dev/null | head -1 | grep -q .; then
        LOG_FILE=$(ls $LOG 2>/dev/null | head -1)
        LOG_FOUND=true

        # Extract progress info
        LAST_STEP=$(grep -oP "Step \K\d+" "$LOG_FILE" 2>/dev/null | tail -1)
        TOTAL_STEPS=$(grep -oP "max_steps=\K\d+" "$LOG_FILE" 2>/dev/null | head -1)

        if [ -z "$TOTAL_STEPS" ]; then
            TOTAL_STEPS="24690"  # Default for our training
        fi

        if [ ! -z "$LAST_STEP" ]; then
            PROGRESS=$(echo "scale=2; $LAST_STEP / $TOTAL_STEPS * 100" | bc)
            echo "  Current Step: $LAST_STEP / $TOTAL_STEPS (${PROGRESS}%)"

            # Estimate time remaining
            RUNTIME=$(ps -p $PID -o etimes= | xargs)
            if [ ! -z "$RUNTIME" ] && [ "$LAST_STEP" -gt 0 ]; then
                TIME_PER_STEP=$(echo "scale=2; $RUNTIME / $LAST_STEP" | bc)
                REMAINING_STEPS=$(echo "$TOTAL_STEPS - $LAST_STEP" | bc)
                REMAINING_TIME=$(echo "scale=0; $REMAINING_STEPS * $TIME_PER_STEP" | bc)

                HOURS=$(echo "$REMAINING_TIME / 3600" | bc)
                MINS=$(echo "($REMAINING_TIME % 3600) / 60" | bc)

                echo "  Estimated remaining: ${HOURS}h ${MINS}m"
            fi
        fi

        # Show recent accuracy/loss
        echo ""
        echo "  Recent metrics:"
        grep -E "(loss|accuracy|eval)" "$LOG_FILE" 2>/dev/null | tail -5 | sed 's/^/    /'

        break
    fi
done

if [ "$LOG_FOUND" = false ]; then
    echo "  ⚠️  No log file found"
    echo "  Training may just be starting..."
fi

echo ""
echo "----------------------------------------------------------------"
echo "Runtime Info:"
echo "----------------------------------------------------------------"

# Process runtime
RUNTIME=$(ps -p $PID -o etimes= | xargs)
if [ ! -z "$RUNTIME" ]; then
    HOURS=$(echo "$RUNTIME / 3600" | bc)
    MINS=$(echo "($RUNTIME % 3600) / 60" | bc)
    SECS=$(echo "$RUNTIME % 60" | bc)
    echo "  Running for: ${HOURS}h ${MINS}m ${SECS}s"
fi

echo ""
echo "================================================================"
echo ""
echo "Commands:"
echo "  Watch in real-time: watch -n 5 $0"
echo "  Attach to tmux:     tmux attach -t byt5_training"
echo "  GPU monitoring:     watch -n 2 nvidia-smi"
echo ""
