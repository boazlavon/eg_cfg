#!/bin/sh -x
export PARTITION=cpu-killable
export GPUS_COUNT=0
export CPUS_COUNT=1
export MODEL_NAME=deepseek-ai/DeepSeek-V3-0324

model_name=$(echo $MODEL_NAME | sed 's_/_\__g')
export RUN="${model_name}"
unique_id=$(date +%Y%m%d_%H%M%S_%N)
mkdir -p "output/$model_name"
export OUTPUT_FILE_PATH="output/$model_name/inference.web.$RUN.$unique_id.out"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

sbatch \
-c $CPUS_COUNT \
--partition=$PARTITION \
--output $OUTPUT_FILE_PATH \
--job-name=$RUN \
scripts/job_runners/slurms/inference.inference_endpoint.slurm

if [ "$1" = "watch" ]; then
    watch tail -n50 $OUTPUT_FILE_PATH
fi
