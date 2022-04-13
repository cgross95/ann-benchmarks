#!/bin/bash
DATASET=siemens-big-sherpa
METRICS="recall total_time"
OPTIONS="--smooth --intervals"
echo "bruteforce flann"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann $OPTIONS
echo "bruteforce hnswlib"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce hnswlib $OPTIONS
echo "bruteforce dciknn pydci"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn pydci $OPTIONS
echo "bruteforce dciknn flann hnswlib pydci sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn flann hnswlib pydci sptag $OPTIONS
echo "bruteforce flann hnswlib sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann hnswlib sptag $OPTIONS
