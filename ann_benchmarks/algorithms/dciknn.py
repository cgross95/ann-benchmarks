from __future__ import absolute_import
import numpy as np
from dciknn import DCI
from ann_benchmarks.algorithms.base import BaseANN


class DCIKNN(BaseANN):
    def __init__(self, dim, num_comp_indices=2, num_simp_indices=7,
                 num_levels=2,
                 construction_field_of_view=10,
                 construction_prop_to_retrieve=0.002):
        self.name = 'DCIKNN(nci=%d, nsi=%d, nl=%d, cfov=%d, cptr=%f)' % (
            num_comp_indices, num_simp_indices, num_levels,
            construction_field_of_view, construction_prop_to_retrieve)
        self._num_levels = num_levels
        self._num_comp_indices = num_comp_indices
        self._num_simp_indices = num_simp_indices
        self._construction_field_of_view = construction_field_of_view
        self._construction_prop_to_retrieve = construction_prop_to_retrieve

        # query arguments
        self._query_field_of_view = None
        self._query_prop_to_retrieve = None

        # set up empty database
        self._dci_db = DCI(dim, self._num_comp_indices,
                           self._num_simp_indices)

    def fit(self, X):
        self._dci_db.add(X, num_levels=self._num_levels,
                         field_of_view=self._construction_field_of_view,
                         prop_to_retrieve=self._construction_prop_to_retrieve)

    def set_query_arguments(self, query_field_of_view, query_prop_to_retrieve):
        self._query_field_of_view = query_field_of_view
        self._query_prop_to_retrieve = query_prop_to_retrieve

    def query(self, v, n):
        return self._dci_db.query(np.array([v]), num_neighbours=n,
                                  field_of_view=self._query_field_of_view,
                                  prop_to_retrieve=self._query_prop_to_retrieve
                                  )[0][0]
