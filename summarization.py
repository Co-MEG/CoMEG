import numpy as np
from scipy import sparse

from sknetwork.topology import get_connected_components


def get_summarized_graph(adjacency: sparse.csr_matrix, patterns: list) -> sparse.csr_matrix:
    """Get summarized graph given patterns and original adjacency matrix. 
       A summarized graph is union of all subgraphs from a list of patterns.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    patterns: list  
        List of tuples where each tuple is an unexpected pattern made of (extent, intent).  

    Returns
    -------
        CSR matrix of the summarized graph.       
    """
    
    rows, cols = [], []

    for c in patterns:

        # exclude first element of lattice 
        if len(c[1]) > 0:
            nodes = sorted(c[0])
            idx = 0
            idx_nodes = np.array([-1] * len(nodes)) # number of unique nodes from patterns
            # reindex nodes
            for n in nodes:
                if n not in idx_nodes:
                    idx_nodes[idx] = n
                    idx += 1
            
            # Record edges from subgraph related to pattern
            adj_pattern = adjacency[nodes, :][:, nodes].tocoo()
            reindex_rows = [int(idx_nodes[src]) for src in adj_pattern.row]
            reindex_cols = [int(idx_nodes[dst]) for dst in adj_pattern.col]
            rows += reindex_rows
            cols += reindex_cols
            
    return sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=adjacency.shape).tocsr()

def get_summarized_biadjacency(adjacency: sparse.csr_matrix, biadjacency: sparse.csr_matrix, patterns: list) -> sparse.csr_matrix:
    """Get summarized biadjacency matrix given an original graph and a list of patterns. Summarized biadjacency contains all links between nodes and attributes that are induced by a summarized graph.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    biadjacency: sparse.csr_matrix
        Biadjacency matrix of the graph
    patterns:  list
        List of tuples where each tuple is an unexpected pattern made of (extent, intent).  

    Returns
    -------
        CSR matrix of the summarized biadjacency matrix.   
    """
    summarized_biadj = np.zeros((adjacency.shape[0], biadjacency.shape[1]))
    for p in patterns:
        if len(p[1]) > 0:
            for node in p[0]:
                summarized_biadj[node, p[1]] += 1

    summarized_biadj = sparse.csr_matrix(summarized_biadj.astype(bool), shape=summarized_biadj.shape)
    
    return summarized_biadj

def get_pattern_summaries(summarized_adjacency: sparse.csr_matrix):
    """Extract connected components from a summarized graph and return labels. Labels are returned only for nodes in a connected component with size > 1.
    
    Parameters
    ----------
    summarized_adjacency: sparse.csr_matrix
        Adjacency matrix of the summarized graph.
        
    Outputs
    -------
        Array of labels, node mask. """
    # Summarized graph filtered on used nodes
    mask = np.flatnonzero(summarized_adjacency.dot(np.ones(summarized_adjacency.shape[1])))
    
    # Number of connected components (NOT considering isolated nodes)
    labels_cc_summarized = get_connected_components(summarized_adjacency[mask, :][:, mask])
    
    return labels_cc_summarized, mask   