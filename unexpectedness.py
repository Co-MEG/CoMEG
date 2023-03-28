import numpy as np
from scipy import sparse, special

from compressors import mdl_graph, mdl_bigraph


def graph_unexpectedness(adjacency, gen_complexities) -> float:
    """Unexpectedness of a graph structure.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph.
    gen_complexities: dict
         Dictionnary with number of nodes as keys and list of graph generation complexities as values.
         
    Outputs
    -------
        Unexpectedness of a graph structure as a float value. """
    n = adjacency.shape[0]
    complexity_desc_g = mdl_graph(adjacency.astype(bool) + sparse.identity(n).astype(bool))
    try:
        avg = np.mean(gen_complexities.get(n))
    except TypeError:
        print(f'Number of nodes missing in dict: {n}')
        print(gen_complexities.keys())
        avg = 0
    
    complexity_gen_g = avg
    return complexity_gen_g - complexity_desc_g

def attr_unexpectedness(biadjacency, attributes, degrees) -> float:
    """Unexpectedness of a list of attributes.
    
    Parameters
    ----------
    biadjacency: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
    attributes: list
        List of attribute indexes.
    degrees: np.ndarray
        Array of attribute degrees in biadjacency
         
    Outputs
    -------
        Unexpectedness of list of attributes as a float value. """
    complexity_gen_a = np.log2(special.comb(biadjacency.shape[1], len(attributes)))
    complexity_desc_a = 0
    for a in attributes:
        complexity_desc_a += np.log2(degrees[a])
    return complexity_gen_a - complexity_desc_a


def attr_unexpectedness_diff(attribute, extent_prev, extent, degrees) -> float:
    cw_a = np.log2(1 / (degrees[attribute] / np.sum(degrees)))  # Generation complexity relates to a number of nodes

    # Description complexity also relates to a number of nodes (actually a difference in number of nodes)
    #cd_a = np.log2((len(extent_prev) - len(extent)) + 1)

    # Description complexity is a ratio between the size of extent of pattern with new attribute and the size of the
    # extent of pattern without this attribute. This ratio is closer to a probability than a difference, which
    # suits more the computation of complexity (?).
    cd_a = np.log2((1 / (len(extent) / len(extent_prev))) + 1)

    print(f'# nodes in X_prev: {len(extent)} - # nodes in X: {len(extent)}')
    print(f'cw_a: {cw_a} - cd_a: {cd_a}')
    return cw_a - cd_a


def attr_unexpectedness_modif(biadjacency, gen_attrs) -> float:
    """Unexpectedness of a list of attributes.

    Parameters
    ----------
    biadjacency: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.

    Outputs
    -------
        Unexpectedness of list of attributes as a float value. """

    n, m = biadjacency.shape
    complexity_desc_a = mdl_bigraph(biadjacency.astype(bool))
    try:
        #avg = np.mean(gen_attrs.get(m))
        med = np.median(gen_attrs.get(m))
    except TypeError:
        print(f'Number of attributes missing in dict: {m}')
        print(gen_attrs.keys())
        #avg = 0
        med = 0

    #complexity_gen_a = avg
    complexity_gen_a = med
    #print(f'Gen comp (avg): {avg}')
    print(f'Gen comp (med): {med}')
    print(f'Comp desc a: {complexity_desc_a}')
    return complexity_gen_a - complexity_desc_a


def pattern_unexpectedness(adjacency, biadjacency, gen_complexities, attributes, degrees) -> float:
    """Pattern unexpectedness, as the sum of the unexpectedness of its elements.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph.
    biadjacency: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
    gen_complexities: dict
        Dictionnary with number of nodes as keys and list of graph generation complexities as values.
    attributes: list
        List of attribute indexes.
    degrees: np.ndarray
        Array of attribute degrees in biadajcency
    
    Outputs
    -------
        Unexpectedness of pattern as a float value. """

    u_g = graph_unexpectedness(adjacency, gen_complexities)
    u_a = attr_unexpectedness(biadjacency, attributes, degrees)
    
    return u_g + u_a
