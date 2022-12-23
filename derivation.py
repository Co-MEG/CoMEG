import numpy as np

from sknetwork.utils import get_neighbors

def intention(nodes, context) -> np.ndarray:
    """Intention of an array of nodes.
    
    Parameters
    ----------
    nodes: np.ndarray
        Array of node indexes
    context: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
        
    Outputs
    -------
        Array of attributes shared by all nodes."""
    if len(nodes) == 0:
        return np.arange(0, context.shape[1])
    intent = get_neighbors(context, node=nodes[0])
    if len(nodes) == 1:
        return intent
    else:
        intent = set(intent)
        for o in nodes[1:]:
            intent &= set(get_neighbors(context, node=o))
            if len(intent) == 0:
                break
        return np.array(list(intent))
    
def extension_csc(attributes, context_csc) -> np.ndarray:
    """Extension of an array of attributes, using CSC format.
    
    Parameters
    ----------
    attributes: np.ndarray
        Array of attribute indexes
    context_csc: sparse.csc_matrix
        Features matrix of the graph in csc format. Contains nodes x attributes.
        
    Outputs
    -------
        Array of nodes sharing all attributes."""
    if len(attributes) == 0:
        return np.arange(0, context_csc.shape[0])
    else:
        res = set(context_csc[:, attributes[0]].indices)
        if len(attributes) > 1:
            for a in attributes[1:]:
                res &= set(context_csc[:, a].indices)
                if len(res) == 0:
                    break
        return np.array(list(res))

def extension(attributes, context):
    """Extension of an array of attributes.
    
    Parameters
    ----------
    attributes: np.ndarray
        Array of attribute indexes
    context: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
        
    Outputs
    -------
        Array of nodes sharing all attributes."""
    ext = get_neighbors(context, node=attributes[0], transpose=True)
    if len(attributes) == 1:
        return ext
    else:
        ext = set(ext)
        for a in attributes[1:]:
            ext &= set(get_neighbors(context, node=a, transpose=True))
            if len(ext) == 0:
                break
        return np.array(list(ext))
                    