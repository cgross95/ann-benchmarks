Dynamic benchmarking nearest neighbors
=====================================

Note
====

This project is a fork of the original [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) repository. We make significant use of their codebase and extend it to handle dynamically updating algorithms. Please see the original project for more information. Attempts have been made to keep the original functionality of the benchmarks accessible, but this project is mainly concerned with dynamically updating algorithms.


Evaluated
=========

The following methods are those evaluated in the original project, with annotations concerning their support for dynamic updates. We have also added `dci-knn` and `pydci`.

| Method | Dynamic updates? | Notes on dynamic updates |
| :----- | :---------------------: | :----------------------- | 
| [Annoy](https://github.com/spotify/annoy) | :x: | "in particular you can not add more items once the tree has been created" |
| [FLANN](http://www.cs.ubc.ca/research/flann/) | :heavy_check_mark: | "Working with dynamic point clouds without a need to rebuild entire kd-tree index" |
| [scikit-learn](http://scikit-learn.org/stable/modules/neighbors.html): LSHForest, KDTree, BallTree | :x: | Requires rebuilding the tree each time |
| [PANNS](https://github.com/ryanrhymes/panns) | :x: | Can be compared to old indices but needs to be recreated |
| [NearPy](http://pixelogik.github.io/NearPy/) | :x: | Rehashes with each new query vector |
| [KGraph](https://github.com/aaalgo/kgraph) | :x: | The index file has to be rebuilt with each new dataset.  |
| [NMSLIB (Non-Metric Space Library)](https://github.com/nmslib/nmslib): SWGraph, HNSW, BallTree, MPLSH | :x: | Static data only |
| [hnswlib (a part of nmslib project)](https://github.com/nmslib/hnsw) | :heavy_check_mark: | Updatable module |
| [RPForest](https://github.com/lyst/rpforest) | :x: | Tree structures have to be rebuilt |
| [FAISS](https://github.com/facebookresearch/faiss.git) | :x: | The x_i's are assumed to be fixed |
| [DolphinnPy](https://github.com/ipsarros/DolphinnPy) | :x: | Hypeplane LSH family model |
| [Datasketch](https://github.com/ekzhu/datasketch) | :x: | Rebuild index each time |
| [PyNNDescent](https://github.com/lmcinnes/pynndescent) | :x: | Rebuild index each time |
| [MRPT](https://github.com/teemupitkanen/mrpt) | :x: | Rebuild index each time |
| [NGT](https://github.com/yahoojapan/NGT): ONNG, PANNG, QG | :x: | Project has evolved into VALD, which supports dynamic updating |
| [SPTAG](https://github.com/microsoft/SPTAG) | :heavy_check_mark: | "Fresh update: Support online vector deletion and insertion" |
| [PUFFINN](https://github.com/puffinn/puffinn) | :x: | Can insert points but needs to be rebuilt after insertion |
| [N2](https://github.com/kakao/n2) | :x: | Rebuild each time, but can be split over several threads |
| [ScaNN](https://github.com/google-research/google-research/tree/master/scann) | :x: | Works well with large data, but does not support dynamic updating |
| [Elastiknn](https://github.com/alexklibisz/elastiknn) | :heavy_check_mark: | "Implementation based on standard Elasticsearch and Lucene primitives, entirely in the JVM. Indexing and querying scale horizontally with Elasticsearch." |
| [OpenSearch KNN](https://github.com/opensearch-project/k-NN) | :x: | High latency in large dimensional vectors |
| [DiskANN](https://github.com/microsoft/diskann): Vamana, Vamana-PQ | :heavy_check_mark: | Updates when points are added |
| [Vespa](https://github.com/vespa-engine/vespa) | :heavy_check_mark: | "Vespa is self-repairing and dynamic" |
| [scipy](https://docs.scipy.org/doc/scipy/reference/spatial.html): cKDTree | :x: | Tree structure needs to be rebuilt |
| [vald](https://github.com/vdaas/vald) | :heavy_check_mark: | "Vald uses distributed index graphs so it continues to work during indexing" |
| [dci-knn](https://github.com/ke-li/dci-knn) | :x: | |
| [pydci](https://github.com/cgross95/pydci) | :heavy_check_mark: | |

Data sets
=========

The datasets used for this project were provided to us by Siemens and is not included in this repository. Before running any tests on this data, please move the following files to the `data` directory: `AS.csv`, `OLHC.csv`, `SHERPA.csv`, `SHERPA_100000.csv`.

Note that the data included with the original project should be able to be adapted for dynamic tests. See the function `ann-benchmarks.datasets.write_dynamic_output` for more information and `ann-benchmarks.datasets.siemens_dynamic` for an example.

Install
=======

The only prerequisite is Python (tested with 3.6) and Docker.

1. Clone the repo.
2. Run `pip install -r requirements.txt`.
3. Run `python install.py` to build all the libraries inside Docker containers (this can take a while, like 10-30 minutes).

Running
=======

In order to replicate the results summarized in the report, try using the script `run_script.sh`. As currently setup, it will take a very long time to run, so running only one dataset or algorithm at a time may be preferable. This can be performed with the command
```
python run.py --dataset $DATASET --count -1 --runs 5 --algorithm $ALGORITHM
```

where `$DATASET` can be any of
- `siemens-sherpa`
- `siemens-big-sherpa`
- `siemens-olhc`
- `siemens-as`
and `$ALGORITHM` can be any of
- `bruteforce`
- `dciknn`
- `flann`
- `hnswlib`
- `pydci`
- `sptag`

See `python run.py --help` for further explanation and command line arguments.

In order to change the values tested in the parameter sweep for each method, change the `args` and `query-args` entries under the desired methods in `algos.yaml`.

Since the testing workflow is somewhat complex, a detailed walkthrough is provided in `docs/dynamic_workflow_outline.md`.

Plotting
========

To replicate the majority of plots in the report, after running `run_script.sh`, run `plot_script.sh`. This script will plot results for a few combinations of algorithms on all datasets. See `python plot_dynamic.py --help` for more information, and see the commands in `plot_script.sh` for examples. Note that specifying metrics in the `--best_metric` argument will find the run with the best behavior in that metric over the last 10% of queries. These algorithms will also be cached separately so that replotting will be much faster. Using the `--force` option will recompute these best runs rather than using their cached versions.

In addition, see `custom_plot_dynamic.py` for plotting individual runs rather than averages of all runs or those best in a certain metric. The main difference is that instead of algorithm names in the `--algorithms` argument, the specific HDF5 files of the desired runs to be plotted should be passed instead.

Including your algorithm
========================

1. Add your algorithm into `ann_benchmarks/algorithms` by providing a small Python wrapper.
2. Add a Dockerfile in `install/` for it
3. Add it to `algos.yaml`

Principles
==========

* Everyone is welcome to submit pull requests with tweaks and changes to how each library is being used.
* In particular: if you are the author of any of these libraries, and you think the benchmark can be improved, consider making the improvement and submitting a pull request.
* This is meant to be an ongoing project and represent the current state.
* Make everything easy to replicate, including installing and preparing the datasets.
* Try many different values of parameters for each library and ignore the points that are not on the precision-performance frontier.
* High-dimensional datasets with approximately 100-1000 dimensions. This is challenging but also realistic. Not more than 1000 dimensions because those problems should probably be solved by doing dimensionality reduction separately.
* Single queries are used by default. ANN-Benchmarks enforces that only one CPU is saturated during experimentation, i.e., no multi-threading. A batch mode is available that provides all queries to the implementations at once. Add the flag `--batch` to `run.py` and `plot.py` to enable batch mode. 
* Avoid extremely costly index building (more than several hours).
* Focus on datasets that fit in RAM. For billion-scale benchmarks, see the related [big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks) project.
* We mainly support CPU-based ANN algorithms. GPU support exists for FAISS, but it has to be compiled with GPU support locally and experiments must be run using the flags `--local --batch`. 
* Do proper train/test set of index data and query points.
* Note that we consider that set similarity datasets are sparse and thus we pass a **sorted** array of integers to algorithms to represent the set of each user.


Authors
=======

Original project built by [Erik Bernhardsson](https://erikbern.com) with significant contributions from [Martin Aumüller](http://itu.dk/people/maau/) and [Alexander Faithfull](https://github.com/ale-f).

The modifications for this project were primarily made by [Craig Gross](https://math.msu.edu/~grosscra) with support from Daniel Ejsmont, Cullen Haselby, and Jim Lewis at Michigan State University.

Related Publication
==================

The following publication details design principles behind the original benchmarking framework: 

- M. Aumüller, E. Bernhardsson, A. Faithfull:
[ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms](https://arxiv.org/abs/1807.05614). Information Systems 2019. DOI: [10.1016/j.is.2019.02.006](https://doi.org/10.1016/j.is.2019.02.006)

Related Projects
================

- [big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks) is a benchmarking effort for billion-scale approximate nearest neighbor search as part of the [NeurIPS'21 Competition track](https://neurips.cc/Conferences/2021/CompetitionTrack).

