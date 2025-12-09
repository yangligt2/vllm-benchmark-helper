#!/bin/bash

# Check if a filename is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <log_file>"
    # Output header for piping convenience
    echo "filename,mean_ms,median_ms,p90_ms,p99_ms,max_ms"
    exit 1
fi

LOG_FILE=$1

grep "get_num_new_matched_tokens -> update_state_after_alloc" "$LOG_FILE" | \
awk '$(NF-1) >= 1 {print $(NF-1)}' | \
sort -n | \
awk -v filename="$LOG_FILE" '
{
    data[NR] = $1;
    sum += $1;
}
END {
    if (NR > 0) {
        count = NR
        mean = sum / count
        p50_index = int(count * 0.5)
        p90_index = int(count * 0.90)
        p99_index = int(count * 0.99)

        if (p50_index == count * 0.5 && count > 1) {
            median = (data[p50_index] + data[p50_index + 1]) / 2
        } else {
            median = data[p50_index + 1]
        }

        p90 = data[p90_index + 1]
        p99 = data[p99_index + 1]
        max = data[count]

        printf "%s,%.3f,%.3f,%.3f,%.3f,%.3f\n", filename, mean, median, p90, p99, max
    } else {
        printf "%s,0,0,0,0,0\n", filename
    }
}'