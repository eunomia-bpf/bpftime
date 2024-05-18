#! /bin/bash

archive=$1
output_dir=$2

mkdir -p "$output_dir"

# List files in the archive
files=$(ar t "$archive")

# extract and renaming each file individually to avoid overriding same object file
for file in $files; do
 ar x "$archive" "$file"

 new_file="$file"
 count=1
 while [[ -f "$output_dir/$new_file" ]]; do
     new_file="${file%.o}_$count.o"
     count=$((count+1))
 done
 mv "$file" "$output_dir/$new_file"
done