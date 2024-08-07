#!/bin/bash

thread_numbers=(4 16)
read_percentages=(0.5)
output_file="result_ycsb.txt"

# Clear the output file
> "$output_file"

for THREAD_NUM in "${thread_numbers[@]}"; do
  for READ_PRE in "${read_percentages[@]}"; do
    for i in {1..2}; do
      echo "Running test FlexiCC with THREAD_NUM=$THREAD_NUM, READ_PRE=$READ_PRE, iteration $i"
      # Run the benchmark script and capture the output
      result=$(bash ./scripts/bench.sh -a -t "$THREAD_NUM" -r "$READ_PRE")

      # Extract the required information
      client=$(echo "$result" | grep -oP 'client:\K[0-9]+')
      success_txn=$(echo "$result" | grep -oP 'success_txn:\K[0-9.]+')
      cc_abort=$(echo "$result" | grep -oP 'cc_abort:\K[0-9.]+')
      ave_latency=$(echo "$result" | grep -oP 'ave_latency:\K[0-9.]+µs')
      p99_latency=$(echo "$result" | grep -oP 'p99_latency:\K[0-9.]+µs')

      # Save the extracted information to the output file
      echo -e "$client\t$success_txn\t$cc_abort\t$ave_latency\t$p99_latency" >> "$output_file"
    done
  done
done

echo "FlexiCC tests completed. Results saved in $output_file."

for THREAD_NUM in "${thread_numbers[@]}"; do
  for READ_PRE in "${read_percentages[@]}"; do
    for i in {1..2}; do
      echo "Running test SSI with THREAD_NUM=$THREAD_NUM, READ_PRE=$READ_PRE, iteration $i"
      # Run the benchmark script and capture the output
      result=$(bash ./scripts/bench_ssi.sh -a -t "$THREAD_NUM" -r "$READ_PRE")

      # Extract the required information
      client=$(echo "$result" | grep -oP 'client:\K[0-9]+')
      success_txn=$(echo "$result" | grep -oP 'success_txn:\K[0-9.]+')
      cc_abort=$(echo "$result" | grep -oP 'cc_abort:\K[0-9.]+')
      ave_latency=$(echo "$result" | grep -oP 'ave_latency:\K[0-9.]+µs')
      p99_latency=$(echo "$result" | grep -oP 'p99_latency:\K[0-9.]+µs')

      # Save the extracted information to the output file
      echo -e "$client\t$success_txn\t$cc_abort\t$ave_latency\t$p99_latency" >> "$output_file"
    done
  done
done

echo "SSI tests completed. Results saved in $output_file."