#!/bin/bash

if [ "$#" -lt 3 ]; then
	echo "usage: ./train_varying.sh <GPU number> <list of p:i pairs> <logdir> <optional extra parameters>"
    echo "p:i pairs = percentage:iterations, pairs being spearated by a space"
	exit
fi

device_nr=$1
percentages=$2
logdir=$3
batch_size=4
sequence_length=16384

shift 3

for p in $percentages; do
    percent=$(echo $p | cut -d ":" -f 1)
    iterations=$(echo $p | cut -d ":" -f 2)
	CUDA_VISIBLE_DEVICES=$device_nr python train.py with batch_size=$batch_size sequence_length=$sequence_length percent_real=$percent iterations=$iterations logdir=$logdir/"$percent"_percent $@
done
