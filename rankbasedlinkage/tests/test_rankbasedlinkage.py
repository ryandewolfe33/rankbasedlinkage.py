import rankbasedlinkage
import scipy.sparse
import numpy as np

import pytest

def test_rankbasedlinkage():
    # Graph taken from Table 1. Weights are 1/ranking to make higher weights appear closer.
    ood = scipy.sparse.csr_matrix([
        [2, 1/7, 1/6, 1/4, 1/8, 1/3, 1/1, 1/5, 1/9, 1/2],
        [1/5, 2, 1/3, 1/6, 1/7, 1/8, 1/1, 1/4, 1/9, 1/2],
        [1/7, 1/6, 2, 1/1, 1/9, 1/5, 1/3, 1/4, 1/8, 1/2],
        [1/2, 1/9, 1/5, 2, 1/4, 1/6, 1/3, 1/8, 1/7, 1/1],
        [1/9, 1/8, 1/7, 1/4, 2, 1/3, 1/6, 1/2, 1/1, 1/5],
        [1/4, 1/9, 1/3, 1/6, 1/5, 2, 1/1, 1/7, 1/8, 1/2],
        [1/1, 1/6, 1/5, 1/4, 1/8, 1/3, 2, 1/7, 1/9, 1/2],
        [1/7, 1/9, 1/1, 1/8, 1/2, 1/3, 1/5, 2, 1/4, 1/6],
        [1/9, 1/6, 1/7, 1/5, 1/1, 1/3, 1/8, 1/2, 2, 1/4],
        [1/7, 1/5, 1/4, 1/2, 1/8, 1/3, 1/1, 1/6, 1/9, 2],
    ])

    L = rankbasedlinkage.rankbasedlinkage.mutual_friend_list(ood)
    linkage_graph = rankbasedlinkage.rankbasedlinkage.in_sway(ood, L)

    answer = scipy.sparse.csr_matrix([
        [0, 2, 2, 5, 0, 5, 8, 1, 0, 2],
        [2, 0, 3, 0, 0, 0, 3, 0, 0, 4],
        [2, 3, 0, 4, 0, 4, 4, 5, 0, 5],
        [5, 0, 4, 0, 2, 3, 5, 1, 1, 7],
        [0, 0, 0, 2, 0, 2, 0, 6, 8, 0],
        [5, 0, 4, 3, 2, 0, 6, 2, 1, 6],
        [8, 3, 4, 5, 0, 6, 0, 0, 0, 7],
        [1, 0, 5, 1, 6, 2, 0, 0, 5, 1],
        [0, 0, 0, 1, 8, 1, 0, 5, 0, 0],
        [2, 4, 5, 7, 0, 6, 7, 1, 0, 0],
    ])

    assert np.array_equal(answer.todense(), linkage_graph.todense())


def test_rankbasedlinkage_class():
    ood = scipy.sparse.csr_matrix([
        [2, 1/7, 1/6, 1/4, 1/8, 1/3, 1/1, 1/5, 1/9, 1/2],
        [1/5, 2, 1/3, 1/6, 1/7, 1/8, 1/1, 1/4, 1/9, 1/2],
        [1/7, 1/6, 2, 1/1, 1/9, 1/5, 1/3, 1/4, 1/8, 1/2],
        [1/2, 1/9, 1/5, 2, 1/4, 1/6, 1/3, 1/8, 1/7, 1/1],
        [1/9, 1/8, 1/7, 1/4, 2, 1/3, 1/6, 1/2, 1/1, 1/5],
        [1/4, 1/9, 1/3, 1/6, 1/5, 2, 1/1, 1/7, 1/8, 1/2],
        [1/1, 1/6, 1/5, 1/4, 1/8, 1/3, 2, 1/7, 1/9, 1/2],
        [1/7, 1/9, 1/1, 1/8, 1/2, 1/3, 1/5, 2, 1/4, 1/6],
        [1/9, 1/6, 1/7, 1/5, 1/1, 1/3, 1/8, 1/2, 2, 1/4],
        [1/7, 1/5, 1/4, 1/2, 1/8, 1/3, 1/1, 1/6, 1/9, 2],
    ])

    rbl = rankbasedlinkage.RankBasedLinkage(n_neighbors=10)
    rbl.fit(ood)

    answer = scipy.sparse.csr_matrix([
            [0, 2, 2, 5, 0, 5, 8, 1, 0, 2],
            [2, 0, 3, 0, 0, 0, 3, 0, 0, 4],
            [2, 3, 0, 4, 0, 4, 4, 5, 0, 5],
            [5, 0, 4, 0, 2, 3, 5, 1, 1, 7],
            [0, 0, 0, 2, 0, 2, 0, 6, 8, 0],
            [5, 0, 4, 3, 2, 0, 6, 2, 1, 6],
            [8, 3, 4, 5, 0, 6, 0, 0, 0, 7],
            [1, 0, 5, 1, 6, 2, 0, 0, 5, 1],
            [0, 0, 0, 1, 8, 1, 0, 5, 0, 0],
            [2, 4, 5, 7, 0, 6, 7, 1, 0, 0],
        ])
    assert np.array_equal(answer.todense(), rbl.linkage.todense())

    clustering = rbl.predict(T="critical", min_cluster_size=1)
    answer = np.array([0, 1, 2, 0, 3, 0, 0, 3, 3, 0])
    assert np.array_equal(answer, clustering)

