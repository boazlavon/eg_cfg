#!/bin/sh -x
#export PARTITION=gpu-h100-killable
#export PARTITION=gpu-ai
export PARTITION=killable
export GPUS_COUNT=1
export CPUS_COUNT=1
export TOKENIZERS_PARALLELISM=false

export MODEL_NAME=deepseek-ai/deepseek-coder-1.3b-instruct
export PROMPT_TYPE=deepseek_instruct

#export MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-instruct
#export PROMPT_TYPE=deepseek_instruct

#export MODEL_NAME=deepseek-ai/deepseek-coder-1.3b-base
#export PROMPT_TYPE=deepseek_base

#export MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
#export PROMPT_TYPE=deepseek_base

#export MODEL_NAME=Qwen/CodeQwen1.5-7B-Chat
#export PROMPT_TYPE=deepseek_instruct

model_name=$(echo $MODEL_NAME | sed 's_/_\__g')
export RUN="${model_name}_${GPUS_COUNT}_gpus"
export PYTHONPATH=$PYTHONPATH:/home/ai_center/ai_users/boazlavon/data/code/dsgi
export PYTHONUNBUFFERED=1
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
