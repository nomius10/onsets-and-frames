#!/bin/bash
if [ "$#" -ne 1 ]; then
	echo "usage: ./getLengthMinutes.sh <path_to_dataset_root>"
	exit
fi
awk '{v=$1/1000/60; printf "%f minutes\n", v}' <(find "$1" -name *.flac | xargs mediainfo --Inform="Audio;%Duration%\n" 2>/dev/null | sed '/^$/d' | paste -sd+ - | bc)
