from __future__ import absolute_import
import numpy as np
from pydci import DCI
from ann_benchmarks.algorithms.base import BaseANN


class PYDCIKNN(BaseANN):
    def __init__(self, dim, num_simple=10, num_composite=10):
        self.name = 'PYDCIKNN(d=%d, num_simple=%d, num_composite=%d' % (
            dim, num_simple, num_composite)
        self._dim = dim
        self._num_simple = num_simple
        self._num_composite = num_composite

        # query arguments
        self._max_retrieve_const = None
        self._max_composite_visit_const = None

        # set up empty database
        self._dci = DCI(dim, num_simple, num_composite)
        self._fitted = False

    def fit(self, X):
        # Will reset if fit multiple times; use update to add points
        if self._fitted:
            self._dci = DCI(self._dim, self._num_simple, self._num_composite,
                            X)
        else:
            self._dci.add(X)
            self._fitted = True

    def update(self, X):
        self._dci.add(X)
        self._fitted = True

    def set_query_arguments(self, max_retrieve_const,
                            max_composite_visit_const):
        self._max_retrieve_const = max_retrieve_const
        self._max_composite_visit_const = max_composite_visit_const

    def query(self, v, n):
        indices, _, _ = self._dci.query(
            np.array([v]), k=n,
            max_retrieve_const=self._max_retrieve_const,
            max_composite_visit_const=self._max_composite_visit_const,
        )
        return indices
