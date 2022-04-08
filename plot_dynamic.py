import os
import copy
import matplotlib as mpl
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.utils import create_linestyles


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
    return np.array(recalls)


def jaccard(dataset_neighbors, alg_neighbors):
    jaccards = []
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
        jaccards.append(len(np.intersect1d(dataset_indices, alg_indices)) /
                        len(np.union1d(dataset_indices, alg_indices)))
    return np.array(jaccards)


def approximation_ratio(data, step, dataset_max_distances, alg_neighbors):
    ratios = []
    for i in range(len(alg_neighbors)):
        query = data[(i + 1) * step]
        try:
            num_alg_indices = np.nonzero(alg_neighbors[i] == -1)[0][0]
            alg_indices = alg_neighbors[i][:num_alg_indices]
        except IndexError:
            alg_indices = alg_neighbors[i]
        if len(alg_indices) == 0:
            max_approx_dist = np.inf
        else:
            # alg_indices.sort()
            # max_approx_dist = np.amax(np.linalg.norm(
            #     data[alg_indices] - query, axis=1))
            # Assumes furthest point is at the end of the list
            max_approx_dist = np.linalg.norm(data[alg_indices[-1]] - query)
        ratios.append(dataset_max_distances[i] / max_approx_dist)
    return np.array(ratios)


def moving_average(a, n=100):
    # From https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        metavar="DATASET",
        default='siemens-sherpa')
    parser.add_argument(
        '--algorithms',
        metavar='ALG',
        nargs='+',
        required=True,
        help='List of algorithms to plot')
    parser.add_argument(
        '--best_metric',
        nargs='+',
        choices=[
            'build_time',
            'query_time',
            'total_time',
            'recall',
            'jaccard',
            'ratio'
        ],
        help='Plot only the result from each algorithm with the best average value (over all query) in this metric')
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Smooth out all plots by taking rolling average'
    )
    parser.add_argument(
        '-o', '--output')
    args = parser.parse_args()

    args.algorithms.sort()
    if args.best_metric:
        args.best_metric = list(set(args.best_metric))
        args.best_metric.sort()

    def plot_mod(t):
        if args.smooth:
            return moving_average(t)
        else:
            return t

    if not args.output:
        if not os.path.isdir('results/dynamic'):
            os.mkdir('results/dynamic')
        args.output = 'results/dynamic/%s_%s' % (args.dataset,
                                                 '_'.join(args.algorithms))
        if args.best_metric:
            args.output += '_' + '_'.join(args.best_metric)
        args.output += '.png'
    results = {}
    for alg in args.algorithms:
        results[alg] = []
        alg_dir = f'results/{args.dataset}/dynamic/{alg}'
        if not os.path.isdir(alg_dir):
            raise OSError(f'No results for algorithm {alg}')
        for result_file in os.scandir(alg_dir):
            results[alg].append(result_file)

    # Load/cache dataset
    dataset, _, _ = get_dataset(args.dataset)
    data = dataset['train']
    dataset_neighbors = dataset['neighbors']
    dataset_distances = np.array(dataset['distances'])
    dataset_max_distances = np.max(
        np.where(dataset_distances < float('inf'), dataset_distances, -np.inf),
        axis=1)

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    alg_metrics = {}
    all_alg_metrics = {}
    if args.best_metric:
        average_alg_metrics = {metric: {} for metric in args.best_metric}
    for alg, result_files in results.items():
        if args.best_metric:
            for average_alg_metric in average_alg_metrics.values():
                average_alg_metric[alg] = {
                    'average': np.inf,  # Assumes lower is better
                    'label': None,
                    'metrics': None
                }
        for result_file in result_files:
            result_name, _ = os.path.splitext(os.path.basename(result_file))
            alg_label = f'{alg}_{result_name}'
            try:
                with h5py.File(result_file, 'r+') as f:
                    alg_metrics['build_time'] = np.array(f['build_times'])
                    alg_metrics['search_time'] = np.array(f['search_times'])
                    alg_metrics['total_time'] = alg_metrics['build_time'] + \
                        alg_metrics['search_time']
                    # Negate recall so lower is better
                    alg_metrics['recall'] = -recall(
                        dataset_neighbors, f['neighbors'])
                    # Negate jaccard so lower is better
                    alg_metrics['jaccard'] = -jaccard(
                        dataset_neighbors, f['neighbors'])
                    # Negate approximation ratio so lower is better
                    alg_metrics['ratio'] = -approximation_ratio(
                        data, dataset.attrs['step'], dataset_max_distances,
                        f['neighbors'])
                    if args.best_metric:
                        for metric in args.best_metric:
                            # Average over last 10% of runs
                            num_average = int(
                                0.1 * len(alg_metrics['build_time']))
                            average = np.mean(
                                alg_metrics[metric][-num_average:])
                            if (average_alg_metrics[metric][alg]['average']
                                    > average):
                                average_alg_metrics[metric][alg]['average'] =\
                                    average
                                average_alg_metrics[metric][alg]['label'] =\
                                    alg_label
                                average_alg_metrics[metric][alg]['metrics'] =\
                                    copy.deepcopy(alg_metrics)
                    else:
                        all_alg_metrics[alg_label] = copy.deepcopy(alg_metrics)
            except OSError as error:
                raise OSError('Check owner of result files.')
    if args.best_metric:
        for metric, average_alg_metric in average_alg_metrics.items():
            for alg_data in average_alg_metric.values():
                alg_label = f"{alg_data['label']}, (best {metric})"
                all_alg_metrics[alg_label] = alg_data['metrics']
    linestyles = create_linestyles(all_alg_metrics.keys())
    for alg_label, alg_metrics in all_alg_metrics.items():
        color, faded, linestyle, marker = linestyles[alg_label]
        axs[0, 0].plot(plot_mod(alg_metrics['build_time']),
                       label=alg_label, color=color, linestyle=linestyle,
                       marker=marker, markevery=0.25, ms=7, lw=3, mew=2)
        axs[0, 1].plot(plot_mod(alg_metrics['search_time']),
                       color=color, linestyle=linestyle, marker=marker,
                       markevery=0.25, ms=7, lw=3, mew=2)
        axs[1, 0].plot(plot_mod(alg_metrics['total_time']),
                       color=color, linestyle=linestyle, marker=marker,
                       markevery=0.25, ms=7, lw=3, mew=2)
        # Make sure to un-negate recall
        axs[1, 1].plot(plot_mod(-alg_metrics['recall']),
                       color=color, linestyle=linestyle, marker=marker,
                       markevery=0.25, ms=7, lw=3, mew=2)
        # Make sure to un-negate approximation ratios
        axs[2, 0].plot(plot_mod(-alg_metrics['ratio']),
                       color=color, linestyle=linestyle, marker=marker,
                       markevery=0.25, ms=7, lw=3, mew=2)
    axs[0, 0].set_ylabel('Time to build index (sec)')
    axs[0, 1].set_ylabel('Time to search (sec)')
    axs[1, 0].set_ylabel('Total time (sec)')
    axs[1, 1].set_ylabel('Recall')
    axs[2, 0].set_ylabel('Approximation ratio')
    for ax_row in axs:
        for ax in ax_row:
            ax.set_xlabel('Iteration Number')
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
            #           prop={'size': 9})
    axs[2, 1].axis('off')
    fig.legend(loc='upper left', bbox_to_anchor=(0.5, 0.27), prop={'size': 10})
    fig.savefig(args.output, bbox_inches='tight')
