#!/bin/bash
METRICS="recall total_time elapsed" 

### BIG SHERPA ###

DATASET=siemens-big-sherpa
OPTIONS="--smooth 100 --landscape --font 20"

# ALL
echo "bruteforce dciknn flann hnswlib pydci sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn flann hnswlib pydci sptag $OPTIONS --build_lim -0.001 6 --search_lim 0 25 --total_lim -0.001 25

# DCI ONLY
echo "bruteforce dciknn pydci"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn pydci $OPTIONS --build_lim -0.001 6 --search_lim 0 25 --total_lim -0.001 25

# NON-DCI ONLY
echo "bruteforce flann hnswlib sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann hnswlib sptag $OPTIONS --build_lim -0.01 0.2 --search_lim -0.001 0.02 --total_lim -0.01 0.2

# HNSW
echo "bruteforce hnswlib"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce hnswlib $OPTIONS --build_lim -0.00005 0.02 --search_lim -0.0005 0.007 --total_lim -0.00005 0.02

# SMOOTH FLANN
echo "bruteforce flann"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann $OPTIONS --build_lim -0.0005 0.01 --search_lim -0.0005 0.007 --total_lim -0.0005 0.011
# # NONSMOOTH FLANN
# echo "bruteforce flann"
# python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann $OPTIONS --build_lim -0.0005 0.5 --search_lim -0.0005 0.007 --total_lim -0.0005 0.5

# DCIKNN
echo "bruteforce dciknn"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn $OPTIONS --build_lim -0.00005 0.02 --search_lim -0.0005 0.007 --total_lim -0.00005 0.02

# PYDCI
echo "bruteforce sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce pydci $OPTIONS --build_lim -0.00005 0.02 --search_lim -0.0005 0.007 --total_lim -0.00005 0.02

# SPTAG
echo "bruteforce sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce sptag $OPTIONS --build_lim -0.00005 0.02 --search_lim -0.0005 0.007 --total_lim -0.00005 0.02

# SHERPA ###

DATASET=siemens-sherpa
OPTIONS="--smooth 10 --landscape --font 20"

# ALL
echo "bruteforce dciknn flann hnswlib pydci sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn flann hnswlib pydci sptag $OPTIONS --build_lim -0.01 1 --search_lim 0 5 --total_lim -0.01 5

# DCI ONLY
echo "bruteforce dciknn pydci"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn pydci $OPTIONS --build_lim -0.001 6 --search_lim 0 25 --total_lim -0.001 25

# NON-DCI ONLY
echo "bruteforce flann hnswlib sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann hnswlib sptag $OPTIONS --build_lim -0.01 0.2 --search_lim -0.001 0.02 --total_lim -0.01 0.2

# HNSW
echo "bruteforce hnswlib"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce hnswlib $OPTIONS --build_lim -0.0005 0.015 --search_lim -0.00005 0.002 --total_lim -0.0005 0.015

# SMOOTH FLANN
echo "bruteforce flann"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann $OPTIONS --build_lim -0.0005 0.004 --search_lim -0.00005 0.002 --total_lim -0.0005 0.004
# # NONSMOOTH FLANN
# echo "bruteforce flann"
# python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann $OPTIONS --build_lim -0.0005 0.5 --search_lim -0.0005 0.007 --total_lim -0.0005 0.5

# DCIKNN
echo "bruteforce dciknn"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn $OPTIONS --build_lim -0.00005 0.3 --search_lim -0.00005 0.01 --total_lim -0.00005 0.3 

# PYDCI
echo "bruteforce sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce pydci $OPTIONS --build_lim -0.00005 0.004 --search_lim -0.00005 3 --total_lim -0.00005 3 --ratio_lim 0 5

# SPTAG
echo "bruteforce sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce sptag $OPTIONS --build_lim -0.00005 0.2  --search_lim -0.00005 0.005 --total_lim -0.00005 0.2


# OLHC ###

DATASET=siemens-olhc
OPTIONS="--smooth 10 --landscape --font 20"

# ALL
echo "bruteforce dciknn flann hnswlib pydci sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn flann hnswlib pydci sptag $OPTIONS --build_lim -0.01 1 --search_lim 0 5 --total_lim -0.01 5

# DCI ONLY
echo "bruteforce dciknn pydci"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn pydci $OPTIONS --build_lim -0.001 6 --search_lim 0 25 --total_lim -0.001 25

# NON-DCI ONLY
echo "bruteforce flann hnswlib sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann hnswlib sptag $OPTIONS --build_lim -0.01 0.2 --search_lim -0.001 0.02 --total_lim -0.01 0.2

# HNSW
echo "bruteforce hnswlib"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce hnswlib $OPTIONS --build_lim -0.0005 0.015 --search_lim -0.00005 0.002 --total_lim -0.0005 0.015

# SMOOTH FLANN
echo "bruteforce flann"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann $OPTIONS --build_lim -0.0005 0.004 --search_lim -0.00005 0.002 --total_lim -0.0005 0.004
# # NONSMOOTH FLANN
# echo "bruteforce flann"
# python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann $OPTIONS --build_lim -0.0005 0.5 --search_lim -0.0005 0.007 --total_lim -0.0005 0.5

# DCIKNN
echo "bruteforce dciknn"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn $OPTIONS --build_lim -0.00005 0.3 --search_lim -0.00005 0.01 --total_lim -0.00005 0.3 

# PYDCI
echo "bruteforce sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce pydci $OPTIONS --build_lim -0.00005 0.004 --search_lim -0.00005 3 --total_lim -0.00005 3 --ratio_lim 0 5

# SPTAG
echo "bruteforce sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce sptag $OPTIONS --build_lim -0.00005 0.2  --search_lim -0.00005 0.005 --total_lim -0.00005 0.2


### AS ###

DATASET=siemens-as
OPTIONS="--smooth 10 --landscape --font 20"

# ALL
echo "bruteforce dciknn flann hnswlib pydci sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn flann hnswlib pydci sptag $OPTIONS --build_lim -0.01 1 --search_lim 0 5 --total_lim -0.01 5

# DCI ONLY
echo "bruteforce dciknn pydci"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn pydci $OPTIONS --build_lim -0.001 6 --search_lim 0 25 --total_lim -0.001 25

# NON-DCI ONLY
echo "bruteforce flann hnswlib sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann hnswlib sptag $OPTIONS --build_lim -0.01 0.2 --search_lim -0.001 0.02 --total_lim -0.01 0.2

# HNSW
echo "bruteforce hnswlib"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce hnswlib $OPTIONS --build_lim -0.0005 0.015 --search_lim -0.00005 0.002 --total_lim -0.0005 0.015

# SMOOTH FLANN
echo "bruteforce flann"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann $OPTIONS --build_lim -0.0005 0.004 --search_lim -0.00005 0.002 --total_lim -0.0005 0.004
# # NONSMOOTH FLANN
# echo "bruteforce flann"
# python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce flann $OPTIONS --build_lim -0.0005 0.5 --search_lim -0.0005 0.007 --total_lim -0.0005 0.5

# DCIKNN
echo "bruteforce dciknn"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce dciknn $OPTIONS --build_lim -0.00005 0.3 --search_lim -0.00005 0.01 --total_lim -0.00005 0.3 

# PYDCI
echo "bruteforce sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce pydci $OPTIONS --build_lim -0.00005 0.004 --search_lim -0.00005 3 --total_lim -0.00005 3 --ratio_lim 0 5

# SPTAG
echo "bruteforce sptag"
python plot_dynamic.py --dataset $DATASET --best_metric $METRICS --algorithms bruteforce sptag $OPTIONS --build_lim -0.00005 0.2  --search_lim -0.00005 0.005 --total_lim -0.00005 0.2
