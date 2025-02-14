from __future__ import absolute_import
import pyflann
import numpy
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN


class FLANN(BaseANN):
    def __init__(self, metric, algorithm, target_precision, branching,
                 iterations, scale_checks=False):
        self._target_precision = target_precision
        self.name = 'FLANN(target_precision=%f)' % self._target_precision
        self._metric = metric
        self._algorithm = algorithm
        self._branching = branching
        self._iterations = iterations
        # scale_checks determines whether checks should be constant (False) or
        # scale with the number of neighbors and branches at each level (True)
        self._scale_checks = scale_checks
        print(self.name)

    def fit(self, X):
        self._flann = pyflann.FLANN(
            target_precision=self._target_precision, algorithm=self._algorithm,
            build_weight=1, branching=self._branching,
            iterations=self._iterations, log_level='info')
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        print(self._flann.build_index(X))

    def update(self, X):
        self._flann.add_points(X)

    def set_query_arguments(self, checks):
        self._checks = checks

    def query(self, v, n):
        if self._metric == 'angular':
            v = sklearn.preprocessing.normalize([v], axis=1, norm='l2')[0]
        # if v.dtype != numpy.float32:
        #     v = v.astype(numpy.float32)
        if self._scale_checks:
            check_val = int((n / self._branching) * self._checks)
        else:
            check_val = self._checks
        return self._flann.nn_index(v, n, checks=check_val)[0][0]
