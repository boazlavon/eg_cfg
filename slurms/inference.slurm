#!/bin/sh
if [ $GPUS_COUNT -gt 1 ]; then
  accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$GPUS_COUNT dsgi.py --start $START_IDX --end $END_IDX --model-name $MODEL_NAME
else
  python dsgi.py --model-name $MODEL_NAME
fi
