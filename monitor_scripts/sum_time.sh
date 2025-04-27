#!/bin/bash

total_seconds=0

# Read already pre-filtered lines
while read -r time_field; do
    # Skip if empty
    if [ -z "$time_field" ]; then
        continue
    fi

    # Check if time is in HH:MM:SS or MM:SS
    if [[ "$time_field" == *:*:* ]]; then
        # Format: HH:MM:SS
        IFS=':' read -r h m s <<< "$time_field"
        seconds=$((10#$h * 3600 + 10#$m * 60 + 10#$s))
    elif [[ "$time_field" == *:* ]]; then
        # Format: MM:SS
        IFS=':' read -r m s <<< "$time_field"
        seconds=$((10#$m * 60 + 10#$s))
    else
        continue
    fi

    total_seconds=$((total_seconds + seconds))
done

# Final output: total seconds
echo "$total_seconds"
