#!/bin/sh -x
#export PARTITION=gpu-h100-killable
export PARTITION=gpu-ai
export GPUS_COUNT=1
export CPUS_COUNT=1
export TOKENIZERS_PARALLELISM=false
export MODEL_NAME=meta-llama/Llama-3.2-1B
#export MODEL_NAME=google/gemma-3-1b-it
model_name=$(echo $MODEL_NAME | sed 's_/_\__g')
export RUN="${model_name}_${GPUS_COUNT}_gpus"
export PYTHONPATH=$PYTHONPATH:/home/ai_center/ai_users/boazlavon/data/code/dsgi
export PYTHONUNBUFFERED=1
export START_IDX=0
export END_IDX=499
export OUTPUT_FILE_PATH="output/inference.$RUN.$PARTITION.out"
sbatch \
-c $CPUS_COUNT \
-G $GPUS_COUNT \
--partition=$PARTITION \
--output $OUTPUT_FILE_PATH \
--job-name=$RUN \
--nodes=1 \
slurms/inference.slurm 
#--nodelist=n-351 \
#--error $OUTPUT_FILE_PATH \
