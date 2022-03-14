import os
import matplotlib as mpl
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
from ann_benchmarks.datasets import get_dataset


def recall(dataset_neighbors, alg_neighbors):
    recalls = []
    for i in range(len(dataset_neighbors)):
        try:
            num_dataset_indices = np.nonzero(dataset_neighbors[i] == -1)[0][0]
            dataset_indices = dataset_neighbors[i][:num_dataset_indices]
        except IndexError:
            dataset_indices = dataset_neighbors[i]
        try:
            num_alg_indices = np.nonzero(alg_neighbors[i] == -1)[0][0]
            alg_indices = alg_neighbors[i][:num_alg_indices]
        except IndexError:
            alg_indices = alg_neighbors[i]
        recalls.append(len(np.intersect1d(dataset_indices,
                       alg_indices)) / len(dataset_indices))
    return recalls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        metavar="DATASET",
        default='siemens-sherpa')
    parser.add_argument(
        '--algorithms',
        metavar='ALG',
        nargs='*',
        help='List of algorithms to plot')
    parser.add_argument(
        '-o', '--output')
    args = parser.parse_args()

    args.algorithms.sort()
    if not args.output:
        if not os.path.isdir('results/dynamic'):
            os.mkdir('results/dynamic')
        args.output = 'results/dynamic/%s_%s.png' % (args.dataset,
                                                     '_'.join(args.algorithms))
    results = {}
    for alg in args.algorithms:
        results[alg] = []
        alg_dir = f'results/{args.dataset}/dynamic/{alg}'
        if not os.path.isdir(alg_dir):
            raise OSError(f'No results for algorithm {alg}')
        for result_file in os.scandir(alg_dir):
            results[alg].append(result_file)

    fig, axs = plt.subplots(4, 1, figsize=(5, 20))
    for alg, result_files in results.items():
        for result_file in result_files:
            result_name, _ = os.path.splitext(os.path.basename(result_file))
            alg_label = f'{alg}_{result_name}'
            try:
                with h5py.File(result_file, 'r+') as f:
                    build_times = np.array(f['build_times'])
                    search_times = np.array(f['search_times'])
                    total_times = build_times + search_times
                    dataset, _ = get_dataset(args.dataset)
                    recalls = recall(dataset['neighbors'], f['neighbors'])
                    axs[0].plot(build_times, label=alg_label)
                    axs[1].plot(search_times, label=alg_label)
                    axs[2].plot(total_times, label=alg_label)
                    axs[3].plot(recalls, label=alg_label)
            except OSError as error:
                raise OSError('Check owner of result files.')
    axs[0].set_ylabel('Time to build index (sec)')
    axs[1].set_ylabel('Time to search (sec)')
    axs[2].set_ylabel('Total time (sec)')
    axs[3].set_ylabel('Recall')
    for ax in axs:
        ax.set_xlabel('Iteration Number')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 9})
    fig.savefig(args.output, bbox_inches='tight')


