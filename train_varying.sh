#!/bin/bash

percentages="0 50 100 20 80 30 70 40 60 10 90"
batch_size=4
sequence_length=16384

if [ "$#" -ne 1 ]; then
	echo "usage: ./train_varying.sh <GPU number>"
	exit
fi

for p in $percentages; do
	CUDA_VISIBLE_DEVICES=$1 python train.py with batch_size=$batch_size sequence_length=$sequence_length percent_real=$p logdir=comtest/"$p"_percent
done
