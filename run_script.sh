#!/bin/bash
DATASETS="siemens-big-sherpa siemens-sherpa siemens-olhc siemens-as"
ALGORITHMS="bruteforce dciknn flann hnswlib pydci sptag"
OPTIONS="--timeout 324000 --force --run-disabled"
for DATASET in $DATASETS; do
	for ALG in $ALGORITHMS; do
		python run.py --dataset $DATASET --count -1 --runs 5 --algorithm $ALG $OPTIONS
	done
done
