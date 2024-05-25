#!/bin/bash

# Check if the correct number of arguments is provided
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <archive> <output_dir>"
  exit 1
fi

archive=$1
output_dir=$2

# Ensure the output directory exists
mkdir -p "$output_dir"

# List files in the archive
files=$(ar t "$archive")

# Declare an associative array to count occurrences of each file
declare -A file_count

# Count occurrences of each file
for file in $files; do
  ((file_count["$file"]++))
done

# Extract files individually, renaming duplicates
for file in "${!file_count[@]}"; do
  count=${file_count["$file"]}
  if [[ $count -gt 1 ]]; then
    for ((i = 1; i <= count; i++)); do
      ar xN $i "$archive" "$file"
      new_file="${file%.o}.$i.o"
      mv "$file" "$output_dir/$new_file"
    done
  else
    ar x "$archive" "$file"
    mv "$file" "$output_dir/$file"
  fi
done