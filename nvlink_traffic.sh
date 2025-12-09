#!/bin/bash

# Interval in seconds
INTERVAL=2
NUM_GPUS=8

echo "Monitoring NVLink throughput for GPUs 0-$(($NUM_GPUS - 1)) every $INTERVAL second(s)... Press Ctrl+C to stop."

# Declare arrays to hold last values for each GPU
declare -a last_rx_arr
declare -a last_tx_arr

# Get initial values
last_ts=$(date +%s.%N)
for (( i=0; i<$NUM_GPUS; i++ )); do
    last_rx_arr[$i]=$(nvidia-smi nvlink -gt d -i $i | awk 'BEGIN{s=0} /Rx:/ {s+=$5} END{print s}')
    last_tx_arr[$i]=$(nvidia-smi nvlink -gt d -i $i | awk 'BEGIN{s=0} /Tx:/ {s+=$5} END{print s}')
done

sleep $INTERVAL

while true; do
    current_ts=$(date +%s.%N)
    # Calculate time delta
    time_delta=$(awk -v last="$last_ts" -v current="$current_ts" 'BEGIN {print current - last}')

    for (( i=0; i<$NUM_GPUS; i++ )); do
        current_rx=$(nvidia-smi nvlink -gt d -i $i | awk 'BEGIN{s=0} /Rx:/ {s+=$5} END{print s}')
        current_tx=$(nvidia-smi nvlink -gt d -i $i | awk 'BEGIN{s=0} /Tx:/ {s+=$5} END{print s}')

        # Calculate traffic delta in KiB
        rx_delta=$(awk -v current="$current_rx" -v last="${last_rx_arr[$i]}" 'BEGIN {print current - last}')
        tx_delta=$(awk -v current="$current_tx" -v last="${last_tx_arr[$i]}" 'BEGIN {print current - last}')

        # Calculate rate in MB/s (KiB/s / 1024)
        rx_rate=$(awk -v delta="$rx_delta" -v time="$time_delta" 'BEGIN {printf "%.2f", delta / time / 1024}')
        tx_rate=$(awk -v delta="$tx_delta" -v time="$time_delta" 'BEGIN {printf "%.2f", delta / time / 1024}')
        
        # Print the results
        printf "GPU %d -> Rx: %8.2f MB/s | Tx: %8.2f MB/s\n" "$i" "$rx_rate" "$tx_rate"

        # Update last values for the next iteration
        last_rx_arr[$i]=$current_rx
        last_tx_arr[$i]=$current_tx
    done

    last_ts=$current_ts
    echo "----------------------------------------------------"
    sleep $INTERVAL
done