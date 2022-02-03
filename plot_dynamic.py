import os
import matplotlib as mpl
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py


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

    fig, axs = plt.subplots(3, 1, figsize=(5, 15))
    for alg, result_files in results.items():
        for result_file in result_files:
            result_name, _ = os.path.splitext(os.path.basename(result_file))
            alg_label = f'{alg}_{result_name}'
            with h5py.File(result_file, 'r') as f:
                build_times = np.array(f['build_times'])
                search_times = np.array(f['search_times'])
                total_times = build_times + search_times
                axs[0].plot(build_times, label=alg_label)
                axs[1].plot(search_times, label=alg_label)
                axs[2].plot(total_times, label=alg_label)
    axs[0].set_ylabel('Time to build index (sec)')
    axs[1].set_ylabel('Time to search (sec)')
    axs[2].set_ylabel('Total time (sec)')
    for ax in axs:
        ax.set_xlabel('Iteration Number')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 9})
    fig.savefig(args.output, bbox_inches='tight')
