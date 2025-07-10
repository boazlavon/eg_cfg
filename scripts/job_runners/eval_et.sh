#!/bin/sh -x

# ------------------------------------------------------------------------------
# Usage examples:
#
# Basic:
# ./scripts/job_runners/eval_et.sh trials/mbpp_eg_cfg/mbpp/deepseek-ai_DeepSeek-V3-0324 trials/mbpp_eg_cfg/mbpp-et/deepseek-ai_DeepSeek-V3-0324 mbpp
# ./scripts/job_runners/eval_et.sh trials/humaneval_eg_cfg/humaneval/deepseek-ai_DeepSeek-V3-0324 trials/humaneval_eg_cfg/humaneval-et/deepseek-ai_DeepSeek-V3-0324 humaneval
#
# With exec host IP/port (for CodeContests + ExecEval only):
# ./scripts/job_runners/eval_et.sh trials/CodeContests_cfg/CodeContests/deepseek-ai_DeepSeek-V3-0324 trials/CodeContests_cfg/CodeContests__ExecEval/deepseek-ai_DeepSeek-V3-0324 CodeContests ExecEval 127.0.0.1 5000
# ------------------------------------------------------------------------------

export PARTITION=cpu-killable
export GPUS_COUNT=0
export CPUS_COUNT=1

# Assign arguments
export INPUT_DIR="$1"
export OUTPUT_DIR="$2"
export DATASET="$3"
export EVAL_TYPE="$4"
export EXEC_EVAL_HOST_IP="$5"
export EXEC_EVAL_HOST_PORT="$6"

# Validate input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: INPUT_DIR '$INPUT_DIR' does not exist or is not a directory."
    exit 1
fi

unique_id=$(date +%Y%m%d_%H%M%S_%N)
export RUN="MBPPET_${unique_id}"
mkdir -p "output/eval_et"
export OUTPUT_FILE_PATH="output/eval_et/eval_et.$unique_id.out"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Build export list
EXTRA_EXPORTS=""
if [ -n "$EVAL_TYPE" ]; then
    EXTRA_EXPORTS=",EVAL_TYPE"
fi

if [ "$DATASET" = "CodeContests" ] && [ "$EVAL_TYPE" = "ExecEval" ]; then
    EXTRA_EXPORTS="$EXTRA_EXPORTS,EXEC_EVAL_HOST_IP,EXEC_EVAL_HOST_PORT"
fi

sbatch \
  -c $CPUS_COUNT \
  --partition=$PARTITION \
  --output "$OUTPUT_FILE_PATH" \
  --job-name="$RUN" \
  --export=ALL,INPUT_DIR,OUTPUT_DIR,DATASET$EXTRA_EXPORTS \
  scripts/job_runners/slurms/eval_et.slurm
