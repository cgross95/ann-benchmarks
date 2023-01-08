#!/bin/bash
DATASET=siemens-big-sherpa
METRICS="recall total_time"
OPTIONS="--build_lim -0.001 0.02 --search_lim -0.001 0.007 --total_lim -0.001 0.02 --landscape --smooth 100 --output test.png"
# echo "bruteforce flann"
# python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann $OPTIONS
echo "bruteforce hnswlib"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce hnswlib $OPTIONS
# echo "bruteforce dciknn pydci"
# python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn pydci $OPTIONS
# echo "bruteforce dciknn flann hnswlib pydci sptag"
# python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn flann hnswlib pydci sptag $OPTIONS
# echo "bruteforce flann hnswlib sptag"
# python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann hnswlib sptag $OPTIONS
