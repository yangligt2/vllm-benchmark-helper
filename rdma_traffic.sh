#!/bin/bash

# Interval in seconds
INTERVAL=2
NUM_NICS=8

echo "Monitoring RDMA throughput for NICs mlx5_0-$(($NUM_NICS - 1)) every $INTERVAL second(s)... Press Ctrl+C to stop."

# Declare arrays to hold last values for each NIC
declare -a last_rx_bytes_arr
declare -a last_tx_bytes_arr

# Get initial values in Bytes
last_ts=$(date +%s.%N)
for (( i=0; i<$NUM_NICS; i++ )); do
    # Read counter (in 4-byte words) and multiply by 4 to get Bytes
    last_rx_bytes_arr[$i]=$(($(cat /sys/class/infiniband/mlx5_$i/ports/1/counters/port_rcv_data) * 4))
    last_tx_bytes_arr[$i]=$(($(cat /sys/class/infiniband/mlx5_$i/ports/1/counters/port_xmit_data) * 4))
done

sleep $INTERVAL

while true; do
    current_ts=$(date +%s.%N)
    # Calculate time delta
    time_delta=$(awk -v last="$last_ts" -v current="$current_ts" 'BEGIN {print current - last}')

    for (( i=0; i<$NUM_NICS; i++ )); do
        current_rx_bytes=$(($(cat /sys/class/infiniband/mlx5_$i/ports/1/counters/port_rcv_data) * 4))
        current_tx_bytes=$(($(cat /sys/class/infiniband/mlx5_$i/ports/1/counters/port_xmit_data) * 4))

        # Calculate traffic delta in Bytes
        rx_delta=$(awk -v current="$current_rx_bytes" -v last="${last_rx_bytes_arr[$i]}" 'BEGIN {print current - last}')
        tx_delta=$(awk -v current="$current_tx_bytes" -v last="${last_tx_bytes_arr[$i]}" 'BEGIN {print current - last}')

        # Calculate rate in MB/s (Bytes/s / 1024 / 1024)
        rx_rate=$(awk -v delta="$rx_delta" -v time="$time_delta" 'BEGIN {printf "%.2f", delta / time / 1024 / 1024}')
        tx_rate=$(awk -v delta="$tx_delta" -v time="$time_delta" 'BEGIN {printf "%.2f", delta / time / 1024 / 1024}')
        
        printf "NIC mlx5_%d -> Rx: %8.2f MB/s | Tx: %8.2f MB/s\n" "$i" "$rx_rate" "$tx_rate"

        last_rx_bytes_arr[$i]=$current_rx_bytes
        last_tx_bytes_arr[$i]=$current_tx_bytes
    done

    last_ts=$current_ts
    echo "----------------------------------------------------"
    sleep $INTERVAL
done