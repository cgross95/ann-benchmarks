from __future__ import absolute_import

import h5py
import json
import os
import re
import traceback


def get_result_filename(dataset=None, count=None, definition=None,
                        query_arguments=None, batch_mode=False):
    d = ['results']
    if dataset:
        d.append(dataset)
    if count:
        d.append(str(count))
    if definition:
        d.append(definition.algorithm + ('-batch' if batch_mode else ''))
        data = definition.arguments + query_arguments
        d.append(re.sub(r'\W+', '_', json.dumps(data, sort_keys=True))
                 .strip('_') + ".hdf5")
    return os.path.join(*d)


def store_results_dynamic(dataset, max_count, definition, query_arguments,
                          attrs, results):
    fn = get_result_filename(
        dataset, 'dynamic', definition, query_arguments, False)
    head, tail = os.path.split(fn)
    if not os.path.isdir(head):
        os.makedirs(head)
    f = h5py.File(fn, 'w')
    for k, v in attrs.items():
        f.attrs[k] = v
    build_times = f.create_dataset('build_times', (len(results),), 'f')
    index_sizes = f.create_dataset('index_sizes', (len(results),), 'f')
    search_times = f.create_dataset('search_times', (len(results),), 'f')
    neighbors = f.create_dataset('neighbors', (len(results), max_count), 'i')
    distances = f.create_dataset('distances', (len(results), max_count), 'f')
    for i, (build_time, index_size, search_time, ds) in enumerate(results):
        build_times[i] = build_time
        index_sizes[i] = index_size
        search_times[i] = search_time
        neighbors[i] = [n for n, d in ds] + [-1] * (max_count - len(ds))
        distances[i] = [d for n, d in ds]\
            + [float('inf')] * (max_count - len(ds))
    f.close()


def store_results(dataset, count, definition, query_arguments, attrs, results,
                  batch):
    fn = get_result_filename(
        dataset, count, definition, query_arguments, batch)
    head, tail = os.path.split(fn)
    if not os.path.isdir(head):
        os.makedirs(head)
    f = h5py.File(fn, 'w')
    for k, v in attrs.items():
        f.attrs[k] = v
    times = f.create_dataset('times', (len(results),), 'f')
    neighbors = f.create_dataset('neighbors', (len(results), count), 'i')
    distances = f.create_dataset('distances', (len(results), count), 'f')
    for i, (time, ds) in enumerate(results):
        times[i] = time
        neighbors[i] = [n for n, d in ds] + [-1] * (count - len(ds))
        distances[i] = [d for n, d in ds] + [float('inf')] * (count - len(ds))
    f.close()


def load_all_results(dataset=None, count=None, batch_mode=False):
    for root, _, files in os.walk(get_result_filename(dataset, count)):
        for fn in files:
            if os.path.splitext(fn)[-1] != '.hdf5':
                continue
            try:
                f = h5py.File(os.path.join(root, fn), 'r+')
                properties = dict(f.attrs)
                if batch_mode != properties['batch_mode']:
                    continue
                yield properties, f
                f.close()
            except:
                print('Was unable to read', fn)
                traceback.print_exc()


def get_unique_algorithms():
    algorithms = set()
    for batch_mode in [False, True]:
        for properties, _ in load_all_results(batch_mode=batch_mode):
            algorithms.add(properties['algo'])
    return algorithms
