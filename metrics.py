import numpy as np
from scipy import sparse


def diversity(pw_distances: np.ndarray, delta: float=0.2) -> float:
    """Diversity, i.e. ratio between number of pairwise distances above threshold and total number of distances. 
    
    Parameters
    ----------
    pw_distances: np.ndarray
        Pairwise distances.
    delta: float (default=0.2)
        Minimumm pairwise distance threshold.
        
    Outputs
    -------
        Diversity. 
    """
    n = pw_distances.shape[0]
    upper = pw_distances[np.triu_indices(n)]
    nb_ps = np.sum(upper > delta)
    
    return nb_ps / len(upper)

def coverage(summarized_adjacency: sparse.csr_matrix) -> float:
    """Node coverage of summarized graph, i.e. ratio between number of nodes in summarized graph and number of nodes in original graph.
    
    Parameters
    ----------
    summarized_adjacency: sparse.csr_matrix
        Adjacency matrix of the summarized graph
    
    Outputs
    -------
        Node coverage. 
    """
    # number of nodes in summarized graph
    n_nodes = len(np.flatnonzero(summarized_adjacency.dot(np.ones(summarized_adjacency.shape[1]))))
    
    # Coverage
    cov = n_nodes / summarized_adjacency.shape[0]
    
    return cov

def conciseness(summarized_adjacency: sparse.csr_matrix, summarized_biadjacency: sparse.csr_matrix) -> float:
    """Conciseness of summarized graph, i.e. sum of the median of node outdegrees and median of number of attributes per node.
    
    Parameters
    ----------
    summarized_adjacency: sparse.csr_matrix
        Adjacency matrix of the summarized graph
    summarized_biadjacency: sparse.csr_matrix
        Biadjacency matrix of the summarized graph (feature matrix)
    
    Outputs
    -------
        Conciseness. 
    """
    n, m = summarized_adjacency.shape[1], summarized_biadjacency.shape[1]

    out_deg_nodes = summarized_adjacency.dot(np.ones(n))
    nb_nodes_ps = np.median(out_deg_nodes[out_deg_nodes > 0])
    out_deg_attrs = summarized_biadjacency.dot(np.ones(m))
    nb_attrs_ps = np.median(out_deg_attrs[out_deg_attrs > 0])
    
    return np.median(nb_nodes_ps) + np.median(nb_attrs_ps)

def information(summarized_adjacency: sparse.csr_matrix, summarized_biadjacency: sparse.csr_matrix, pw_distances: np.ndarray) -> float:
    """Information of summarized graph, i.e. (diversity x corevage) / conciseness.
    
    Parameters
    ----------
    summarized_adjacency: sparse.csr_matrix
        Adjacency matrix of the summarized graph
    summarized_biadjacency: sparse.csr_matrix
        Biadjacency matrix of the summarized graph (feature matrix)
    pw_distances: np.ndarray
        Pairwise distances.
        
    Outputs
    -------
        Summarized graph information.
    """
    div = diversity(pw_distances)
    cov = coverage(summarized_adjacency)
    conc = conciseness(summarized_adjacency, summarized_biadjacency) 
    information = (div * cov) / np.sqrt(conc)
    print(f'inf: {information*100} - div: {div} - cov: {cov} - conc: {conc}')

    return information * 100