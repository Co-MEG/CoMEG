import numpy as np
import pickle
from scipy import sparse

from sknetwork.data import load_netset
from sknetwork.utils import get_degrees

from utils import get_root_directory

def load_data(dataset: str):
    """Load data and return loaded elements as a tuple.
    
    Parameters
    ----------
    dataset: str
        Name of dataset (on netset or local).
    """
    netset = ['wikivitals-fr', 'wikischools', 'wikivitals', 'wikihumans']
    labels = ''

    if dataset in netset:
        graph = load_netset(dataset)
        if dataset != 'wikihumans':
            labels = graph.labels

    else:
        ROOT_DIR = get_root_directory()
        with open(f'{ROOT_DIR}/data/{dataset}Graph', 'br') as f:
            graph = pickle.load(f)

    adjacency = graph.adjacency
    biadjacency = graph.biadjacency
    names = graph.names
    names_col = graph.names_col
    
    return adjacency, biadjacency, names, names_col, labels

def preprocess_data(biadjacency: sparse.csr_matrix, names_col: np.ndarray, s: int, sort_data=True):
    """Filter and sort features according to support s.
    
    Parameters
    ----------
    biadjacency: sparse.csr_matrix
        Feature matrix of the graph.
    names_col: np.ndarray
        Feature names.
    s: int
        Minimum support for number of attributes.
    sort_data: bool (default=True)
        If True, sort attribute columns according to attribute frequency.

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
    if sort_data:
        sort_index = np.argsort(freq_attribute)
        sorted_biadjacency = biadjacency[:, sort_index]
        words = words[sort_index]
    else:
        sorted_biadjacency = biadjacency.copy()

    return sorted_biadjacency, words

def load_patterns(dataset: str, beta: int, s: int, order: bool, inpath: str, with_prob: bool, delta=None) -> list:
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
    if with_prob:
        if delta is not None:
            with open(f"{inpath}/result_{dataset}_{beta}_{s}_order{str(order)}_delta_{delta}.bin", "rb") as data:
                patterns = pickle.load(data)
        else:
            with open(f"{inpath}/result_{dataset}_{beta}_{s}_order{str(order)}_delta_0.bin", "rb") as data:
                patterns = pickle.load(data)
    else:
        with open(f"{inpath}/result_{dataset}_{beta}_{s}_order{str(order)}.bin", "rb") as data:
            patterns = pickle.load(data)

    return patterns

def get_pw_distance_matrix(dataset: str, beta: int, s: int, path: str, method: str='summaries', delta=None) -> np.ndarray:
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
    if delta is not None:
        with open(f'{path}/wasserstein_distances_{dataset}_{beta}_{s}_delta_{delta}_{method}.pkl', 'rb') as data:
            pw_distances = np.load(data)
    else:
        with open(f'{path}/wasserstein_distances_{dataset}_{beta}_{s}_{method}.pkl', 'rb') as data:
            pw_distances = np.load(data)
    
    return pw_distances

def read_parameters(filename: str):
    """Read parameters from parameter file.
    
    Parameters
    ----------
    filename: str
        Parameter filename """

    parameters = {}

    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    
    for l in lines:
        name = l.split(':')[0]
        values = l.split(':')[1].split(',')
        if name == 'datasets':
            parameters[name] = [v.strip() for v in values]
        elif name == 's':
            parameters[name] = [int(v.strip()) for v in values]
        elif name == 'patterns_path':
            parameters[name] = values[0].strip()
        elif name == 'gamma':
            parameters[name] = float(values[0].strip())
        else:
            raise ValueError(f'{name} is not a valid parameter.')
    return parameters

def get_sias_pattern(pattern: dict, names=None, names_col=None):
    # get subgraph
    if names is not None:
        subgraph_nodes = np.asarray(list(map(int, [np.where(names==x)[0][0] for x in pattern.get('subgraph') if x in names])))
    else:
        subgraph_nodes = np.asarray(list(map(int, pattern.get('subgraph'))))
    
    # get attributes
    pos_attrs = set(pattern.get('characteristic').get('positiveAttributes'))
    neg_attrs = set(pattern.get('characteristic').get('negativeAttributes'))
    
    attrs_list = []
    for x in pos_attrs.union(neg_attrs):
        if '>=' in x:
            attr_value = x.split('>=')[0]
        elif '<=' in x:
            attr_value = x.split('<=')[0]
        else:
            attr_value = x
        attrs_list.append(attr_value)
    if names_col is not None:
        attrs = np.asarray(list(map(int, [np.where(names_col==x)[0][0] for x in attrs_list if x in names_col])))
    else:
        attrs = np.asarray(list(map(int, attrs_list)))
        #attrs = attrs_list.copy()
        #attrs = np.asarray([int(x.split('>=')[0]) for x in pos_attrs.union(neg_attrs)])
    return subgraph_nodes, attrs
    
def get_excess_pattern(pattern: dict, names, names_col):
    
    # get subgraph
    try:
        subgraph_nodes = np.asarray([np.where(names == x)[0][0] for x in pattern.get('subgraph') if '?' not in x])
    except IndexError:
        print(pattern.get('subgraph'))
    
    # get attributes
    pos_attrs = set(pattern.get('characteristic').get('positiveAttributes'))
    neg_attrs = set(pattern.get('characteristic').get('negativeAttributes'))
    attrs = np.asarray([np.where(names_col == x)[0][0] for x in pos_attrs.union(neg_attrs)])
    
    return subgraph_nodes, attrs