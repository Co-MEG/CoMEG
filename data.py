
import numpy as np
import pickle
from scipy import sparse

from sknetwork.data import load_netset
from sknetwork.utils import get_degrees

def load_data(dataset: str):
    """Load data from netset and return loaded elements as a tuple.
    
    Parameters
    ----------
    dataset: str
        Name of dataset on netset.
    """
    graph = load_netset(dataset)
    adjacency = graph.adjacency
    biadjacency = graph.biadjacency
    names = graph.names
    names_col = graph.names_col
    labels = graph.labels
    
    return adjacency, biadjacency, names, names_col, labels

def preprocess_data(biadjacency: sparse.csr_matrix, names_col: np.ndarray, s: int):
    """Filter and sort features according to support s.
    
    Parameters
    ----------
    biadjacency: sparse.csr_matrix
        Feature matrix of the graph.
    names_col: np.ndarray
        Feature names.
    s: int
        Minimum support for number of attributes.

    Outputs
    -------
        Preprocessed feature matrix and names array.
    """
    # Frequent attributes
    freq_attribute = get_degrees(biadjacency.astype(bool), transpose=True)
    index = np.flatnonzero((freq_attribute <= 1000) & (freq_attribute >= s))

    # Filter data with index
    biadjacency = biadjacency[:, index]
    words = names_col[index]
    freq_attribute = freq_attribute[index]
    
    # Sort data
    sort_index = np.argsort(freq_attribute)
    sorted_biadjacency = biadjacency[:, sort_index]
    words = words[sort_index]

    return sorted_biadjacency, words

def load_patterns(dataset: str, beta: int, s: int, order: bool) -> list:
    """Load patterns.
    
    Parameters
    ----------
    dataset: str
        Name of dataset on netset.
    beta: int
        Minimum support value for intent.
    s: int
        Minimum support value for extent.
    order: bool
        Ordering of attributes.
        
    Ouptuts
    -------
        List of patterns. 
    """
    with open(f"output/result/result_{dataset}_{beta}_{s}_order{str(order)}.bin", "rb") as data:
        patterns = pickle.load(data)
    
    return patterns

def get_pw_distance_matrix(dataset: str, beta: int, s: int, method: str='summaries') -> np.ndarray:
    """Load distances matrices.
    
    Parameters
    ----------
    dataset: str:
        Name of dataset on netset.
    beta: int
        Minimum support value for intent.
    s: int
        Minimum support value for extent.
    method: str
        Name of baseline method.
        
    Outputs
    -------
        Matrix of pairwise distances.
    """
    with open(f'output/result/wasserstein_distances_{dataset}_{beta}_{s}_{method}.pkl', 'rb') as data:
        pw_distances = np.load(data)
    
    return pw_distances