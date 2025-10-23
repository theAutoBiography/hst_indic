#!/bin/bash
#
# Run all doping experiments sequentially
# This will train models on each dataset and collect results
#

set -e

echo "=========================================="
echo "RUNNING ALL DOPING EXPERIMENTS"
echo "=========================================="
echo ""

# Base directory for experiments
EXP_DIR="data/experiments/doping"

# List of experiments to run
EXPERIMENTS=(
    "baseline_00pct"
    "malayalam_doping_05pct"
    "malayalam_doping_10pct"
    "malayalam_doping_20pct"
    "malayalam_doping_30pct"
    "malayalam_doping_50pct"
)

# Optional: Add Telugu experiments if they have sufficient data
# "telugu_doping_05pct"
# "telugu_doping_10pct"

RESULTS_FILE="doping_experiments_results.txt"

echo "Starting experiments at $(date)" > $RESULTS_FILE
echo "=========================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "EXPERIMENT: $exp"
    echo "=========================================="
    echo ""

    # Run training
    python scripts/train_doping_experiment.py "$EXP_DIR/$exp"

    # Extract results
    RESULT_FILE="models/byt5_${exp}/test_results.json"

    if [ -f "$RESULT_FILE" ]; then
        ACCURACY=$(python -c "import json; print(json.load(open('$RESULT_FILE'))['test_exact_match_accuracy'])")
        echo "$exp: $ACCURACY%" >> $RESULTS_FILE
        echo "✓ Completed: $exp - Accuracy: $ACCURACY%"
    else
        echo "✗ Results not found for $exp"
    fi

    echo ""
done

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Results summary:"
cat $RESULTS_FILE
echo ""
echo "Detailed results in: $RESULTS_FILE"
