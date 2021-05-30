#!/usr/bin/env bash

target_dir="./data/jpegs_256/*"
min_num=10000
max_num=-1
find $target_dir -type d -print0 | while read -d '' -r dir; do
	files=("$dir"/*)
	file_num=${#files[@]}
	# printf "%5d files in directory %s\n" "$file_num" "$dir"
	if [ $min_num -gt $file_num ]; then
		min_num=$file_num
		echo "Updated min number of files: $min_num"
	fi
	if [ $max_num -lt $file_num ]; then
		max_num=$file_num
		echo "Updated max number of files: $max_num"
	fi
done
# echo "Max number of files: $max_num; Min number of files: $min_num"
