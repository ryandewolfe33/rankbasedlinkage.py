import scipy.sparse
import numpy as np
import sknetwork as sn

from tqdm import tqdm


def mutual_friend_list(ood: scipy.sparse.csr_matrix):
    L = list()
    for x in tqdm(range(ood.shape[0])):
        x_friends = ood[x].nonzero()[1]
        if len(x_friends) == 0:
            break
        for z in x_friends:
            if ood[z, x] != 0 and x < z:
                L.append([int(x), int(z)])
    return L


def in_sway(ood: scipy.sparse.csr_matrix, L):
    # Linkage graph is same vertices, non-negative integer edge weights
    row_ind = np.empty(len(L), dtype="int32")
    col_ind = np.empty(len(L), dtype="int32")
    data = np.empty(len(L), dtype="int32")

    ood_csc = ood.tocsc()

    for i in tqdm(range(len(L))):
        x, z = L[i]

        # get ys less similar than z to x
        y_to_x = ood_csc[:, x].nonzero()[0]
        x_ranks_y = ood[x, y_to_x]
        xy_beats_xz = x_ranks_y.indices[x_ranks_y.data > ood[x,z]]

        # get ys less similar than x to z
        y_to_z = ood_csc[:, z].nonzero()[0]
        z_ranks_y = ood[z, y_to_z]
        zy_beats_xz = z_ranks_y.indices[z_ranks_y.data > ood[z,x]]

        # In sway calculation
        in_sway = len(np.union1d(y_to_x, y_to_z)) - len(np.union1d(xy_beats_xz, zy_beats_xz))

        row_ind[i] = x
        col_ind[i] = z
        data[i] = in_sway

        """
        for y in ys:
            
            if(
                (ood[x, y] == 0 and ood[y, x] == 0)  # xy not in U_gamma
                or (ood[y, z] == 0 and ood[z, y]==0)  # zy not in U_gamma
            ):
                continue
            if(
                ood[y, x] == 0 and ood[y, z] == 0  # Gamma(y) intersect {x,z} is empty
                or y == x
                or y == z
            ):
                continue
            

            xz_is_source = (
                (ood[x,y] == 0 or ood[x,z] > ood[x,y])
                and (ood[z,y] == 0 or ood[z,x] > ood[z,y])
            )
            if xz_is_source:
                linkage_graph[x,z] += 1
                linkage_graph[z,x] += 1
        """
    
    linkage_graph = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=ood.shape)
    linkage_graph = linkage_graph + linkage_graph.transpose()

    return linkage_graph
        

class RankBasedLinkage():
    """
    An unsupervised clustering algorithm for comparator data.
    """

    def __init__(self, n_neighbors=15):
        self.n_neighbors=15

    
    def fit(self, ood: scipy.sparse.csr_matrix):
        self.ood = ood
        print("Making Friend List")
        L = mutual_friend_list(ood)
        print("Compute In Sway")
        linkage = in_sway(ood, L)
        print("Done")
        self.linkage = linkage
        return self


    def predict(self, T="critical", min_cluster_size=15):
        if T == "critical":
            T = 0
            while np.sum(self.linkage.data >= T)/2 >= self.linkage.shape[0]:
                T += 1

        pruned = self.linkage.copy()
        pruned.data[pruned.data < T] = 0
        pruned.eliminate_zeros()

        clusters = sn.topology.get_connected_components(pruned)
        # Prune clusters and reindex to 0-N
        cluster_sizes = np.zeros(np.max(clusters)+1)
        for c in clusters:
            cluster_sizes[c] += 1
        cluster_fate = np.where(cluster_sizes < min_cluster_size, -1, cluster_sizes)
        next_id = 0
        for i in range(len(cluster_fate)):
            if cluster_fate[i] != -1:
                cluster_fate[i] = next_id
                next_id += 1
        
        for i in range(len(clusters)):
            clusters[i] = cluster_fate[clusters[i]]
        
        return clusters

            