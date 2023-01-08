import os
import copy
import matplotlib as mpl
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
from ann_benchmarks.results import get_best_metric_filename, store_best_metric, get_metrics
from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.utils import create_linestyles
import pandas as pd

metric_dict = {
    'build_time': 'build time',
    'search_time': 'search time',
    'total_time': 'total time',
    'elapsed': 'elapsed time',
    'recall': 'recall',
    'jaccard': 'Jaccard index',
    'ratio': 'approx. ratio'
}


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


def rolling_average(t, n=100):
    df = pd.Series(t)
    return df.rolling(n)


def plot_data(ax, mean, std, flip, linestyle_info, alg_label, smooth,
              intervals):
    color, faded, linestyle, marker = linestyle_info
    if smooth:
        rolling = rolling_average(mean, int(smooth))
        plot_data = rolling.mean().dropna()
        plot_std = pd.Series(std[int(smooth) - 1:], index=plot_data.index)
    else:
        plot_data = pd.Series(mean)
        plot_std = pd.Series(std)
    if flip:
        plot_data = -plot_data
    ax.plot(plot_data, label=alg_label, color=color,
            linestyle=linestyle, marker=marker, markevery=0.25, ms=14, lw=6,
            mew=4)
    low = plot_data - 2 * plot_std
    up = plot_data + 2 * plot_std
    ax.fill_between(plot_data.index, low, up, color=faded)
    # if smooth and intervals:
    #     deviation = rolling.std().dropna()
    #     low = (plot_data - 2 * deviation)
    #     up = (plot_data + 2 * deviation)
    #     ax.fill_between(deviation.index, low, up, color=faded)


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
        choices=metric_dict.keys(),
        help='Plot only the result from each algorithm with the best average value (over all query) in this metric')
    parser.add_argument(
        '--smooth',
        metavar='N',
        default=None,
        # action='store_true',
        help='Smooth out all plots by taking rolling average of length N'
    )
    parser.add_argument(
        '--intervals',
        action='store_true',
        help='Plot intervals if smoothing method supports it'
    )
    parser.add_argument(
        '--show_args',
        action='store_true',
        help='Show arguments of method found in each best metric'
    )
    parser.add_argument(
        '--landscape',
        action='store_true',
        help='Lay out subplots in landscape'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Recompute best metric cache'
    )
    parser.add_argument(
        '--print_elapsed',
        action='store_true',
        help='Print total elapsed time'
    )
    parser.add_argument(
        '--build_lim',
        nargs=2,
        metavar=('lower', 'upper'),
        type=float,
        help='Set limits for vertical axis on build time plot'
    )
    parser.add_argument(
        '--search_lim',
        nargs=2,
        metavar=('lower', 'upper'),
        type=float,
        help='Set limits for vertical axis on search time plot'
    )
    parser.add_argument(
        '--total_lim',
        nargs=2,
        metavar=('lower', 'upper'),
        type=float,
        help='Set limits for vertical axis on total time plot'
    )
    parser.add_argument(
        '--recall_lim',
        nargs=2,
        metavar=('lower', 'upper'),
        type=float,
        help='Set limits for vertical axis on recall plot'
    )
    parser.add_argument(
        '--ratio_lim',
        nargs=2,
        metavar=('lower', 'upper'),
        type=float,
        help='Set limits for vertical axis on ratio plot'
    )
    parser.add_argument(
        '-o', '--output')
    args = parser.parse_args()

    args.algorithms.sort()
    if args.best_metric:
        args.best_metric = list(set(args.best_metric))
        args.best_metric.sort()

    if not args.output:
        if not os.path.isdir('results/dynamic'):
            os.mkdir('results/dynamic')
        args.output = 'results/dynamic/%s_%s' % (args.dataset,
                                                 '_'.join(args.algorithms))
        if args.best_metric:
            args.output += '_' + '_'.join(args.best_metric)
        if args.smooth:
            args.output += '_' + 'smooth'
        if args.intervals:
            args.output += '_' + 'intervals'
        if args.landscape:
            args.output += '_' + 'landscape'
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

    if args.landscape:
        fig, axs = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    else:
        fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)
    axs = axs.flatten()
    alg_metrics_means = {}
    alg_metrics_stds = {}
    all_alg_metrics_means = {}
    all_alg_metrics_stds = {}
    if args.best_metric:
        average_alg_metrics = {metric: {} for metric in args.best_metric}
        non_cached_metrics = {alg: [] for alg in results.keys()}
    for alg, result_dirs in results.items():
        if args.best_metric:
            for metric, average_alg_metric in average_alg_metrics.items():
                best_metric_file = get_best_metric_filename(
                    args.dataset, alg, metric)
                if os.access(best_metric_file, os.R_OK) and not args.force:
                    with h5py.File(best_metric_file, 'r+') as f:
                        average_alg_metric[alg] = {
                            'average': f.attrs['average'],
                            'label': f.attrs['label'],
                            'alg': alg,
                            'metrics_means': get_metrics(f, metric_dict.keys(), 'means'),
                            'metrics_stds': get_metrics(f, metric_dict.keys(), 'stds')
                        }
                else:
                    average_alg_metric[alg] = {
                        'average': np.inf,  # Assumes lower is better
                        'label': None,
                        'alg': alg,
                        'metrics_means': None,
                        'metrics_stds': None
                    }
                    non_cached_metrics[alg].append(metric)
            if not non_cached_metrics[alg]:  # Can skip this algorithm
                continue
        for result_dir in result_dirs:
            if not os.path.isdir(result_dir):
                continue
            result_name = os.path.basename(result_dir)
            alg_label = f'{alg}_{result_name}'
            runs = os.scandir(result_dir)
            runs_metrics = []
            for run_num, result_file in enumerate(runs):
                run_metrics = {}
                try:
                    with h5py.File(result_file, 'r+') as f:
                        run_metrics['build_time'] = np.array(
                            f['build_times'])
                        run_metrics['search_time'] = np.array(
                            f['search_times'])
                        run_metrics['total_time'] = run_metrics['build_time'] + \
                            run_metrics['search_time']
                        run_metrics['elapsed'] = np.sum(
                            run_metrics['total_time'])
                        # Negate recall so lower is better
                        run_metrics['recall'] = -recall(
                            dataset_neighbors, f['neighbors'])
                        # Negate jaccard so lower is better
                        run_metrics['jaccard'] = -jaccard(
                            dataset_neighbors, f['neighbors'])
                        # Negate approximation ratio so lower is better
                        run_metrics['ratio'] = -approximation_ratio(
                            data, dataset.attrs['step'], dataset_max_distances,
                            f['neighbors'])
                        runs_metrics.append(run_metrics)
                except OSError as error:
                    raise OSError('Check owner of result files.')
            for metric in metric_dict.keys():
                runs_df = pd.DataFrame([runs_metrics[i][metric]
                                       for i in range(len(runs_metrics))])
                alg_metrics_means[metric] = runs_df.mean(axis=0)
                alg_metrics_stds[metric] = runs_df.std(axis=0)
            if args.best_metric:
                for metric in non_cached_metrics[alg]:
                    # print(alg_metrics_means)
                    # print(alg_metrics_means[metric])
                    # Average over last 10% of runs
                    num_average = int(
                        0.1 * len(alg_metrics_means['build_time']))
                    average = np.mean(
                        alg_metrics_means[metric][-num_average:])
                    if (average_alg_metrics[metric][alg]['average']
                            > average):
                        average_alg_metrics[metric][alg]['average'] =\
                            average
                        average_alg_metrics[metric][alg]['label'] =\
                            alg_label
                        average_alg_metrics[metric][alg]['metrics_means'] =\
                            copy.deepcopy(alg_metrics_means)
                        average_alg_metrics[metric][alg]['metrics_stds'] =\
                            copy.deepcopy(alg_metrics_stds)
            else:
                all_alg_metrics_means[alg_label] = copy.deepcopy(
                    alg_metrics_means)
                all_alg_metrics_stds[alg_label] = copy.deepcopy(
                    alg_metrics_stds)
    if args.best_metric:
        # Cache new metrics
        for alg, metrics in non_cached_metrics.items():
            for metric in metrics:
                store_best_metric(args.dataset, alg, metric,
                                  average_alg_metrics[metric][alg],
                                  metric_dict.keys())
        for metric, average_alg_metric in average_alg_metrics.items():
            for alg_data in average_alg_metric.values():
                if args.show_args:
                    alg_label =\
                        f"{alg_data['label']}, best {metric_dict[metric]}, elapsed time = {alg_data['metrics_means']['elapsed'][0]:.3f} sec"
                else:
                    alg_label =\
                        f"{alg_data['alg']}, best {metric_dict[metric]}"
                all_alg_metrics_means[alg_label] = alg_data['metrics_means']
                all_alg_metrics_stds[alg_label] = alg_data['metrics_stds']
    linestyles = create_linestyles(all_alg_metrics_means.keys())
    for alg_label in all_alg_metrics_means.keys():
        plot_metrics = [('build_time', False),
                        ('search_time', False),
                        ('total_time', False),
                        ('recall', True),
                        ('ratio', True)]
        for i, (metric, flip) in enumerate(plot_metrics):
            label = alg_label if i == 0 else None
            plot_data(axs[i], all_alg_metrics_means[alg_label][metric],
                      all_alg_metrics_stds[alg_label][metric], flip,
                      linestyles[alg_label], label, args.smooth,
                      args.intervals)
        if args.print_elapsed:
            print(f'{alg_label}, {all_alg_metrics_means[alg_label]["elapsed"]}')
    axs[0].set_ylabel('Time to build index (sec)', size=24)
    axs[0].set_title("(a)", y=0, pad=-45, size=24, verticalalignment="top")
    if args.build_lim:
        axs[0].set_ylim(args.build_lim)
    axs[1].set_ylabel('Time to search (sec)', size=24)
    axs[1].set_title("(b)", y=0, pad=-45, size=24, verticalalignment="top")
    if args.search_lim:
        axs[1].set_ylim(args.search_lim)
    axs[2].set_ylabel('Total time (sec)', size=24)
    axs[2].set_title("(c)", y=0, pad=-45, size=24, verticalalignment="top")
    if args.total_lim:
        axs[2].set_ylim(args.total_lim)
    axs[3].set_ylabel('Recall', size=24)
    axs[3].set_title("(d)", y=0, pad=-45, size=24, verticalalignment="top")
    if args.recall_lim:
        axs[3].set_ylim(args.recall_lim)
    else:
        axs[3].set_ylim([0, 1.1])
    axs[4].set_ylabel('Approximation ratio', size=24)
    axs[4].set_title("(e)", y=0, pad=-45, size=24, verticalalignment="top")
    if args.ratio_lim:
        axs[4].set_ylim(args.ratio_lim)
    else:
        axs[4].set_ylim([0, 1.1])
    for ax in axs:
        # for ax in ax_row:
        ax.set_xlabel('Iteration Number', size=24)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
        #           prop={'size': 9})
    axs[5].axis('off')
    if args.landscape:
        # fig.legend(loc='upper left', bbox_to_anchor=(0.56, 0.41),
        #            prop={'size': 10})
        fig.legend(loc='lower right', prop={'size': 24})
    else:
        fig.legend(loc='upper left', bbox_to_anchor=(0.5, 0.27),
                   prop={'size': 10})
    fig.savefig(args.output, bbox_inches='tight')
