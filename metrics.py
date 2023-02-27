import numpy as np
from scipy import sparse

from sknetwork.utils import get_degrees


def diversity(pw_distances: np.ndarray, gamma: float=0.2) -> float:
    """Diversity, i.e. ratio between number of pairwise distances above threshold and total number of distances. 
    
    Parameters
    ----------
    pw_distances: np.ndarray
        Pairwise distances.
    gamma: float (default=0.2)
        Minimumm pairwise distance threshold.
        
    Outputs
    -------
        Diversity. 
    """
    n = pw_distances.shape[0]
    upper = pw_distances[np.triu_indices(n)]
    nb_ps = np.sum(upper > gamma)
    
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

def coverage_excess(patterns: list, n: int) -> float:
    """Node coverage for Excess algorithm patterns.
    
    Parameters
    ----------
    patterns: list
        List of Excess patterns
    n: int
        Total number of nodes in initial attributed graph. 
        
    Output
    ------
        Node coverage. 
    """
    all_nodes_excess = set()
    for p in patterns:
        all_nodes_excess |= set(p[0])

    cov = len(all_nodes_excess) / n

    return cov

def conciseness(adjacency, biadjacency, summarized_adjacency: sparse.csr_matrix, summarized_biadjacency: sparse.csr_matrix, nb_cc) -> float:
    """Conciseness of summarized graph, i.e. sum of the median of number of nodes and median of number of attributes per pattern/community.
    
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
    nb_nodes_ps = out_deg_nodes[out_deg_nodes > 0]
    out_deg_attrs = summarized_biadjacency.dot(np.ones(m))
    nb_attrs_ps = out_deg_attrs[out_deg_attrs > 0]

    avg_deg = np.mean(get_degrees(adjacency.astype(bool)))
    avg_deg_attr = np.mean(get_degrees(biadjacency.astype(bool)))
    print(f'Avg subgraph size: {np.mean(nb_nodes_ps)} - avg node degree: {avg_deg}')
    print(f'Avg subgraph attributes: {np.mean(nb_attrs_ps)} - avg nb attr: {avg_deg_attr}')
    return (np.mean(nb_nodes_ps) / avg_deg * np.mean(nb_attrs_ps) / avg_deg_attr)

def conciseness_summaries(adjacency, biadjacency, nb_cc, labels_cc_summarized, concept_summarized_attributes) -> float:
    """Conciseness of summarized graph, i.e. sum of the median of number of nodes and median of number of attributes per pattern/community.
    
    Outputs
    -------
        Conciseness. 
    """
    extent_size_summaries, intent_size_summaries = [], []
    print(f'Nb pattern summaries: {nb_cc}')
    for i in range(nb_cc):
        mask_cc = labels_cc_summarized == i
        extent_size_summaries.append(mask_cc.sum())
        intent_size_summaries.append(len(np.flatnonzero(concept_summarized_attributes[i, :])))

    avg_deg = np.mean(get_degrees(adjacency.astype(bool)))
    avg_deg_attr = np.mean(get_degrees(biadjacency.astype(bool)))
    print(f'Avg subgraph size: {np.mean(extent_size_summaries)} - avg node degree: {avg_deg}')
    print(f'Avg subgraph attributes: {np.mean(intent_size_summaries)} - avg nb attr: {avg_deg_attr}')
    #return nb_cc * (np.mean(extent_size_summaries) / avg_deg) * (np.mean(intent_size_summaries) / avg_deg_attr)
    print(f"# patterns: {nb_cc}")
    return nb_cc * np.sqrt(np.mean(extent_size_summaries) * np.mean(intent_size_summaries))

def conciseness_summaries_new(labels_cc_summarized, pattern_summarized_attributes, nb_cc, pattern_summaries=None):
    
    nb_nodes_ps = np.mean([np.sum(labels_cc_summarized==i) for i in range(nb_cc)])
    nb_attrs_ps = np.mean(pattern_summarized_attributes.sum(axis=1))

    # NEW
    if pattern_summaries is not None:
        nodes_ps, attrs_ps = [], []
        for ps in pattern_summaries:
            nodes_ps.append(len(ps[0]))
            attrs_ps.append(len(ps[1]))
        nb_nodes_ps = np.mean(nodes_ps)
        nb_attrs_ps = np.mean(attrs_ps)
        nb_cc = len(pattern_summaries)

    print(f'Avg subgraph size: {nb_nodes_ps}')
    print(f'Avg subgraph attributes: {nb_attrs_ps}')
    print(f"# patterns: {nb_cc}")

    return nb_cc * np.sqrt(nb_nodes_ps * nb_attrs_ps)

def width_excess(patterns_excess) -> float:
    """Width for Excess patterns.
    
    Parameters
    ----------
    patterns_excess: list
        List of Excess patterns.

    Outputs
    -------
        Width. 
    """
    nb_nodes, nb_attrs = [], []
    for i in patterns_excess:
        nb_nodes.append(len(i[0]))
        nb_attrs.append(len(i[1]))

    nb_e_patterns = len(patterns_excess)
    print(f'# patterns: {nb_e_patterns}')
    
    return nb_e_patterns * np.sqrt(np.mean(nb_nodes) * np.mean(nb_attrs))

def information(adjacency, biadjacency, summarized_adjacency: sparse.csr_matrix, summarized_biadjacency: sparse.csr_matrix, nb_cc, pw_distances: np.ndarray, dataset, b, s, method, inpath: str) -> float:
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
    conc = conciseness(adjacency, biadjacency, summarized_adjacency, summarized_biadjacency, nb_cc) 
    information = (div * cov) / (nb_cc * (conc))
    print(f'inf: {information} - div: {div} - cov: {cov} - conc: {nb_cc * (conc)}')

    with open(f'{inpath}/new/information_details_{dataset}_{b}_{s}_{method}_new_conc.txt', 'w') as f:
        f.write(f'{div}, {cov}, {nb_cc * (conc)}, {information}')

    return information 

def information_summaries(adjacency, biadjacency, summarized_adjacency: sparse.csr_matrix, labels_cc_summarized, nb_cc, concept_summarized_attributes, pw_distances: np.ndarray, dataset, b, s, gamma, method, inpath: str, pattern_summaries=None) -> float:
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
    div = diversity(pw_distances, gamma)
    cov = coverage(summarized_adjacency)
    conc = conciseness_summaries_new(labels_cc_summarized, concept_summarized_attributes, nb_cc, pattern_summaries=pattern_summaries)
    
    information = (div * cov) / (conc)
    print(f'inf: {information} - div: {div} - cov: {cov} - conc: {(conc)}')

    with open(f'{inpath}/information_details_{dataset}_{b}_{s}_{method}_{gamma}_new_conc.txt', 'w') as f:
        f.write(f'{div}, {cov}, {(conc)}, {information}')

    return information 