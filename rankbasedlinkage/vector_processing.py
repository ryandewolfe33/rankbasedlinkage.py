import pynndescent
import scipy.sparse
import numpy as np


def build_knn_rank_graph(data, n_neighbors=50, metric="euclidean"):
    index = pynndescent.NNDescent(data, n_neighbors=n_neighbors, metric=metric).neighbor_graph[0]  # Index 1 is distances

    indices = index.flatten()
    indptr = np.arange(0, n_neighbors*index.shape[0]+1, n_neighbors, dtype="int32")

    data = np.tile(np.arange(1, n_neighbors+1, dtype="int32")[::-1], index.shape[0])  # High weigh edges ranked closer

    rank_graph = scipy.sparse.csr_matrix((data, indices, indptr), dtype="int32")
    return rank_graph
