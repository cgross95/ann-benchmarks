# Installation

- Run `python install.py`
	- Helpful command line arguments:
		- `--proc` to use more workers for Docker install
		- `--algorithm` to specify only one Docker image


# Running

## Main command
- Run `python run.py`
	- Helpful command line arguments:
		- See `python run.py --help`
		- Lots of ways to select only certain algorithms/adjust dataset/adjust $k$
		- The most important option for running dynamically is `--count -1`. This initializes a dynamic updating run with the $k$ value to vary as specified in dynamic dataset setup.

## Dynamic workflow

- `run.py`
	- `ann_benchmarks.main`
		- `ann_benchmarks.datasets.get_dataset`
			- Creates HD5 dataset 
			- Loads raw data (see, e.g., `ann_benchmarks.datasets.siemens_dynamic`)
				- `ann_benchmarks.datasets.write_dynamic_output`
					- Using given step size and percentage of nearest neighbors, take raw data and split into batches of training data and query points at the end of each batch
					- Run brute force to create ground truth nearest neighbors
		- `ann_benchmarks.algorithms.definitions.get_definition`
			- Read `algos.yaml` for algorithm information
		- Do a bunch of filtering based on command line arguments and availability of algorithms to get final list of algorithms in `definitions`
		- Create workers to run all the algorithms using:
		- `ann_benchmarks.runner.run_docker`
			- Parses algorithm definition to set up arguments, mount Docker volumes, and set up logging
			- Run established docker container via:

- `run_algorithm.py`
	- `ann_benchmarks.runner.run_from_cmdline`
		- `ann_benchmarks.runner.dynamic`
			- `ann_benchmarks.algorithms.definitions.instantiate_algorithm`
				- Imports specified algorithm and returns instantiated `BaseANN` object
			- Gets dataset, all training data, and attributes
			- Sets individual arguments for queries from `query-args` in `algos.yaml`
			- `ann_benchmarks.runner.run_individual_query_dynamic`
				- Fits training data in batches specified by `step`
				- Chooses query from training data at the end of each batch
				- Set `count` (number of nearest neighbors) to be a percentage (specified by `radius`) of number of points in the search index
				- Different workflow if the algorithm needs to prepare for a query
				- Time to add each batch of points and time for each query are reported separately
			- `ann_benchmarks.results.store_results_dynamic`
				- Stores all build times, search times, neighbors, distances, and other metadata in HD5 file named by dataset, algorithm, and arguments
