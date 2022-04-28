#!/bin/bash
DATASET=siemens-as
ALGORITHMS="bruteforce dciknn flann hnswlib pydci sptag"
OPTIONS="--timeout 324000 --force --run-disabled"
for ALG in $ALGORITHMS; do
	python run.py --dataset $DATASET --count -1 --runs 5 --algorithm $ALG $OPTIONS
done
