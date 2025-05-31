#!/bin/sh -x
export PARTITION=cpu-killable
export GPUS_COUNT=0
export CPUS_COUNT=1

# Usage validation
# if [ "$#" -ne 3 ]; then
#     echo "Usage: $0 <INPUT_DIR> <OUTPUT_DIR> <NUM_WORKERS>"
#       exit 1
# fi

# Assign arguments
export INPUT_DIR="$1"
export OUTPUT_DIR="$2"
export NUM_WORKERS="$3"

# Validate input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: INPUT_DIR '$INPUT_DIR' does not exist or is not a directory."
      exit 1
fi

# Optionally validate NUM_WORKERS is a positive integer
if ! echo "$NUM_WORKERS" | grep -Eq '^[1-9][0-9]*$'; then
    echo "❌ Error: NUM_WORKERS must be a positive integer."
    exit 1
fi

unique_id=$(date +%Y%m%d_%H%M%S_%N)
export RUN="MBPPET_${unique_id}"
mkdir -p "output/mbpp_et"
export OUTPUT_FILE_PATH="output/mbpp_et/mbpp_et.$unique_id.out"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

sbatch \
-c $CPUS_COUNT \
--partition=$PARTITION \
--output $OUTPUT_FILE_PATH \
--job-name=$RUN \
--export=ALL,INPUT_DIR=$INPUT_DIR,OUTPUT_DIR=$OUTPUT_DIR,NUM_WORKERS=$NUM_WORKERS \
scripts/job_runners/slurms/mbpp_et.slurm

if [ "${4:-}" = "watch" ]; then
    watch tail -n50 "$OUTPUT_FILE_PATH"
fi
