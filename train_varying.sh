#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "usage: ./train_varying.sh <GPU number> <list of pr:ps:i pairs> <logdir> <optional extra parameters>"
    echo "p:i pairs = percentagereal:percentagesynth:iterations, pairs being spearated by a space"
    exit
fi

device_nr=$1
percentages=$2
logdir=$3
batch_size=8
sequence_length=32768

shift 3

for p in $percentages; do
    percent=$(echo $p | cut -d ":" -f 1)
    percent2=$(echo $p | cut -d ":" -f 2)
    iterations=$(echo $p | cut -d ":" -f 3)
    CUDA_VISIBLE_DEVICES=$device_nr python train.py with batch_size=$batch_size sequence_length=$sequence_length percent_real=$percent percent_synth=$percent2 iterations=$iterations logdir=$logdir/"$percent"_"$percent2"_percent $@
done