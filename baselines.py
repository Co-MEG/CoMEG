import numpy as np
from scipy import sparse

from sknetwork.clustering import Louvain


# Louvain
resolutions = {'wikivitals-fr': {1: 0, 3: 0, 5: 0.3, 6: 0.4, 10: 0.8, 12: 0.9, 15: 1.1, 47: 3, 27: 1.89}, 
                'wikivitals': {4: 0.5, 7: 0.8, 11: 1, 13: 1.3, 14: 1.18, 16: 1.5, 19: 1.6, 21: 1.76, 29: 2.4, 31: 2.5, 34: 2.7, 47: 3.903, 49: 3.95, 65: 4.70, 101: 6.4}, 
                'wikischools': {4: 0.45, 8: 1, 3: 0.4, 8: 0.58, 9: 0.9, 10: 1.18, 13: 1.4, 15: 1.5, 18: 1.69, 22: 1.95, 39: 2.75, 40: 2.75,  46: 2.99, 58: 3.7, 73: 4.41}}


def get_louvain(dataset: str, adjacency: sparse.csr_matrix, nb_cc: int) -> np.ndarray:
    """Louvain algorithm for clustering graphs by maximization of modularity. Returns labels of the nodes.
    
    Parameters
    ----------
    dataset: str
        Name of dataset on netset.
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    nb_cc: int
        Number of communities
    
    Outputs
    -------
        Array of node labels.
    """
    louvain = Louvain(resolution=resolutions.get(dataset).get(nb_cc)) 
    labels_louvain = louvain.fit_transform(adjacency)
    nb_louvain = len(np.unique(labels_louvain))
    
    return labels_louvain

def get_community_graph(adjacency: sparse.csr_matrix, labels_communities: np.ndarray) -> sparse.csr_matrix:
    """Equivalent of summarized graph but for community-based methods. Returns the adjacency matrix of the graph made of the union of all communities. 
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    labels_communities: np.ndarray
        Array of node community labels 

    Output
    ------
        Sparse matrix of the community graph.
    """
    n_com = len(np.unique(labels_communities))
    rows, cols = [], []
    for n in range(n_com):
        nodes = np.flatnonzero(labels_communities == n)
        idx = 0
        idx_nodes = np.array([-1] * len(nodes)) # number of unique nodes from communities
        # reindex nodes
        for n in nodes:
            if n not in idx_nodes:
                idx_nodes[idx] = n
                idx += 1

        # Record edges from subgraph related to community
        adj_com = adjacency[nodes, :][:, nodes].tocoo()
        reindex_rows = [int(idx_nodes[src]) for src in adj_com.row]
        reindex_cols = [int(idx_nodes[dst]) for dst in adj_com.col]
        rows += reindex_rows
        cols += reindex_cols
        
    return sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=adjacency.shape).tocsr()