# Installation

- Run `python install.py`
	- Helpful command line arguments:
		- `--proc` to use more workers for Docker install
		- `--algorithm` to specify only one Docker image
			- Maybe we start with just sklearn to get a feel for things


# Running

## Main command
- Run `python run.py`
	- Helpful command line arguments:
		- See `python run.py --help`
		- Lots of ways to select only certain algorithms/adjust dataset/adjust $k$

## Workflow

- `run.py`
	- `ann_benchmarks.main`
		- `ann_benchmarks.datasets.get_dataset`
			- Loads/downloads requested dataset in [HD5](https://docs.h5py.org/en/stable/) format
				- To create HD5 dataset:
					- Download raw data
					- Train-test split via `sklearn.model_selection.train_test_split`
					- `ann_benchmarks.datasets.write_output`
						- Create HD5 file with `train`, `test`, `neighbors`, and `distances` datasets (the latter two are created via `ann_benchmarks.algorithms.bruteforce`)
		- `ann_benchmarks.algorithms.definitions.get_definition`
			- Read `algos.yaml` for algorithm information
		- Do a bunch of filtering based on command line arguments and availability of algorithms to get final list of algorithms in `definitions`
		- Create workers to run all the algorithms using:
		- `ann_benchmarks.runner.run_docker`
			- Parses algorithm definition to set up arguments, mount Docker volumes, and set up logging
			- Run established docker container via:

- `run_algorithm.py`
	- `ann_benchmark.runner.run_from_cmdline`
		- `ann_benchmark.runner.run`
			- `ann_benchmark.algorithms.definitions.instantiate_algorithm`
				- Imports specified algorithm and returns instantiated `BaseANN` object
			- Gets train and test data
			- Fits training data
			- Sets individual arguments for queries from `query-args` in `algos.yaml`
			- `ann_benchmark.runner.run_individual_query`
