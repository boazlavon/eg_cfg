#!/bin/sh
# Run the Python script
set -x
echo " Starting:"
echo "   Input dir  : $INPUT_DIR"
echo "   Output dir : $OUTPUT_DIR"
echo "   Threads    : $NUM_WORKERS"
echo

python mbpp_analysis/scripts/mbpp_to_mbpp_et.py "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --trial-workers $NUM_WORKERS
