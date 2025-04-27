#!/bin/bash

# Set the maximum number of jobs you want in the queue
N=50  # Change this to your desired limit
TIME_LOG_FILE="max_job_time_seconds.txt"  # File to track the maximum time

while true; do
    # Get the current number of jobs (excluding the header line)
    n=$(squeue --me | tail -n +2 | wc -l)

    echo "[$(date)] Current jobs: $n / $N"

    # Calculate how many jobs to submit
    if [ "$n" -lt "$N" ]; then
        num_to_submit=$((N - n))
        echo "Submitting $num_to_submit jobs..."

        for ((i = 0; i < num_to_submit; i++)); do
            ./cmdlines/inference_sbatch.sh
            sleep 1  # Optional: small delay between submissions
        done
    else
        echo "No need to submit new jobs."
    fi

    # --- NEW PART: calculate max job time only for deepseek jobs ---
    max_seconds=0

    while read -r line; do
        job_name=$(echo "$line" | awk '{print $3}')
        if [[ "$job_name" != "deepseek" ]]; then
            continue
        fi

        time_field=$(echo "$line" | awk '{print $6}')

        if [[ "$time_field" == *:*:* ]]; then
            IFS=':' read -r h m s <<< "$time_field"
            seconds=$((10#$h * 3600 + 10#$m * 60 + 10#$s))
        elif [[ "$time_field" == *:* ]]; then
            IFS=':' read -r m s <<< "$time_field"
            seconds=$((10#$m * 60 + 10#$s))
        else
            continue
        fi

        if [ "$seconds" -gt "$max_seconds" ]; then
            max_seconds=$seconds
        fi
    done < <(squeue --me | tail -n +2)

    # Read current max from file if it exists
    if [ -f "$TIME_LOG_FILE" ]; then
        current_max=$(cat "$TIME_LOG_FILE")
    else
        current_max=0
    fi

    # If new max is greater, update the file
    if [ "$max_seconds" -gt "$current_max" ]; then
        echo "$max_seconds" > "$TIME_LOG_FILE"
        echo "Updated max time: $max_seconds seconds (was $current_max)"
    else
        echo "No update needed. Current file max: $current_max seconds, this loop max: $max_seconds seconds."
    fi

    # --- End of NEW PART ---

    sleep 60
done
