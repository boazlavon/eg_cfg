#!/bin/sh -x
set -x

echo " Starting:"
echo "   Input dir        : $INPUT_DIR"
echo "   Output dir       : $OUTPUT_DIR"
echo "   Dataset          : $DATASET"

if [ -n "$EVAL_TYPE" ]; then
    echo "   Eval type        : $EVAL_TYPE"
else
    echo "   Eval type        : (default: eval-eg-cfg)"
fi

if [ "$DATASET" = "CodeContests" ] && [ "$EVAL_TYPE" = "ExecEval" ]; then
    echo "   Host IP (opt)    : $EXEC_EVAL_HOST_IP"
    echo "   Host Port (opt)  : $EXEC_EVAL_HOST_PORT"
fi

echo

python eg_cfg/eval_et.py "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --dataset "$DATASET" \
  ${EVAL_TYPE:+--eval-type "$EVAL_TYPE"} \
  ${EXEC_EVAL_HOST_IP:+--exec-eval-host-ip "$EXEC_EVAL_HOST_IP"} \
  ${EXEC_EVAL_HOST_PORT:+--exec-eval-host-port "$EXEC_EVAL_HOST_PORT"}
