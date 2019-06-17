#!/bin/bash

set -e

files=$(find ../Datasets_trained/ft_* -name *.pt)

for model in $files; do
	logfile="ft_tests/$(echo $model | cut -d '/' -f 3- | sed 's/pt/txt/g')"
	logdir=$(echo $logfile | cut -d '/' -f -3)
	#echo "mkdir -p $logdir"
	#echo "testing model $model and saving log to $logfile"

	mkdir -p $logdir
	python evaluate.py $model MAESTRO test | tee "$logfile.m"
	python evaluate.py $model | tee $logfile
done
