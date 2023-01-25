import numpy as np
from scipy import sparse

from sknetwork.data import load_netset

from data import load_data

def dataset2json(dataset: str, outpath: str):
    """Save sknetwork dataset to SIAS-Miner JSON format.
    
    Parameters
    ----------
    dataset: str
        Dataset name
    outpath: str
        Output path
    """
        
    adjacency, biadjacency, names, names_col, labels = load_data(dataset)

    outdict = {}

    outdict['descriptorName'] = dataset
    outdict['attributeName'] = list(names_col)

    
    n = adjacency.shape[0]
    adjacency_lil = adjacency.tolil()

    # Edges
    edges = [{"vertexId": names[u], "connected_vertices": list(names[adjacency_lil.rows[u]])} for u in range(n)]
    

