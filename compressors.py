import numpy as np
from scipy import special


def mdl_graph(adjacency) -> float:
    """Minimum description length for graph structure.
    
    Parameters
    ----------
    adjacency: sparse.csr_matric
        Adjacency matrix of the graph
        
    Outputs
    -------
        Minimum description length of graph structure."""
    n = adjacency.shape[0]
    m = adjacency.nnz

    if n == 0:
        return 0
    else:
        # nodes
        nodes_mdl = np.log2(n + 1)
        
        # edges
        degrees = adjacency.dot(np.ones(n))
        max_degree = np.max(degrees)
        edges_mdl = (n + 1) * np.log2(max_degree + 1) + np.sum([np.log2(special.comb(n, deg)) for deg in degrees])
        #edges_mdl = np.log2(m+1)

        return nodes_mdl + edges_mdl