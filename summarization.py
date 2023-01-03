import numpy as np
from scipy import sparse


def get_summarized_graph(adjacency, patterns) -> sparse.coo_matrix:
    """Get summarized graph given patterns and original adjacency matrix. 
       A summarized graph is union of all subgraphs from a list of patterns.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    patterns:  List of tuples where each tuple is an unexpected pattern made of (extent, intent).  

    Returns
    -------
        COO matrix of the summarized graph.       
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