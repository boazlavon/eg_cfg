#!/bin/sh -x
export PARTITION=killable
export GPUS_COUNT=1
export CPUS_COUNT=1
export MODEL_NAME=deepseek-ai/deepseek-coder-1.3b-instruct

model_name=$(echo $MODEL_NAME | sed 's_/_\__g')
export RUN="${model_name}"
unique_id=$(date +%Y%m%d_%H%M%S_%N)
mkdir -p "output/$model_name"
export OUTPUT_FILE_PATH="output/$model_name/inference.web.$RUN.$unique_id.out"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

sbatch \
-c $CPUS_COUNT \
-G $GPUS_COUNT \
--partition=$PARTITION \
--output $OUTPUT_FILE_PATH \
--job-name=$RUN \
--nodes=1 \
scripts/job_runners/slurms/inference.local.slurm

if [ "$1" = "watch" ]; then
    watch tail -n50 $OUTPUT_FILE_PATH
fi
