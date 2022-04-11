#!/bin/bash
 echo "bruteforce flann"
 python plot_dynamic.py --dataset siemens-big-sherpa --best_metric recall total_time --algorithms bruteforce flann --smooth --intervals --landscape
echo "bruteforce hnswlib"
python plot_dynamic.py --dataset siemens-big-sherpa --best_metric recall total_time --algorithms bruteforce hnswlib --smooth --intervals --landscape
echo "bruteforce dciknn pydci"
python plot_dynamic.py --dataset siemens-big-sherpa --best_metric recall total_time --algorithms bruteforce dciknn pydci --smooth --intervals --landscape
echo "bruteforce dciknn flann hnswlib pydci sptag"
python plot_dynamic.py --dataset siemens-big-sherpa --best_metric recall total_time --algorithms bruteforce dciknn flann hnswlib pydci sptag --smooth --intervals --landscape
echo "bruteforce flann hnswlib sptag"
python plot_dynamic.py --dataset siemens-big-sherpa --best_metric recall total_time --algorithms bruteforce flann hnswlib sptag --smooth --intervals --landscape
