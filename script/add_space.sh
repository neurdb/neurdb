#!/bin/bash

# Define the input and output files
input_file="sql_features.txt"
output_file="sql_features_modified.txt"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Input file '$input_file' not found!"
    exit 1
fi

# Process each line in the input file
while IFS= read -r line || [[ -n "$line" ]]; do
    # Use a regex to match lines ending with "YES" or "NO" and add a space
    if [[ $line =~ (YES|NO)$ ]]; then
        echo "$line\t" >> "$output_file"
    else
        echo "$line" >> "$output_file"
    fi
done < "$input_file"

echo "Processing complete. Modified file saved as '$output_file'."
