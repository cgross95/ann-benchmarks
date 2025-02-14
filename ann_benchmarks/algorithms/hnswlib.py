from __future__ import absolute_import
import os
import hnswlib
import numpy as np
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN


class HnswLib(BaseANN):
    def __init__(self, dim, metric, max_elements=None, method_param=None):
        self.metric = {'angular': 'cosine', 'euclidean': 'l2'}[metric]
        self.method_param = method_param
        self.max_elements = max_elements
        # print(self.method_param,save_index,query_param)
        # self.ef=query_param['ef']
        self.name = 'hnswlib (%s)' % (self.method_param)
        self.dim = dim

    def fit(self, X):
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=self.dim)
        if self.max_elements is None:
            self.max_elements = len(X)
        self.p.init_index(max_elements=self.max_elements,
                          ef_construction=self.method_param["efConstruction"],
                          M=self.method_param["M"])
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)

    def update(self, X):
        data_labels = self.p.get_current_count() + np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)

    def query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]

    def freeIndex(self):
        del self.p
