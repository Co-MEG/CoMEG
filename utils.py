import gensim
import os
import numpy as np
from scipy import sparse

from sknetwork.topology import CoreDecomposition
from sknetwork.utils import directed2undirected

from experiments.pattern_information import MyCorpus


def pattern_attributes(biadjacency: sparse.csr_matrix, labels: np.ndarray, mask=None):
    """Build pattern x attributes matrix. Column values are count of occurrences of attributes for each pattern/community.
    
    Parameters
    ----------
    biadjacency: sparse.csr_matrix
        Biadjacency matrix of the graph
    labels: np.ndarray
        Belonging community for each node in the graph, e.g Louvain labels or KMeans labels
    mask: np.ndarray (default=None)
        Mask for nodes in connected components
        
    Outputs
    -------
        Matrix with patterns/communities in rows and count of attributes in columns. """

    nb_cc = len(np.unique(labels))
    matrix = np.zeros((nb_cc, biadjacency.shape[1]))
    for c in range(nb_cc):
        mask_cc = labels == c
        if mask is not None:
            indices_attr = np.unique(biadjacency[mask, :][mask_cc, :].indices)
        else:
            indices_attr = np.unique(biadjacency[mask_cc, :].indices)
        for ind in indices_attr:
            matrix[c, ind] += 1

    return matrix

def get_s_pattern_attributes(pattern_summaries: list, m: int) -> np.ndarray:
    """Build pattern summaries x attributes matrix. 
    
    Parameters
    ----------
    pattern_summaries: list
        List of pattern summaries as tuples.
    m: int
        Number of attributes in original data.
        
    Output
    ------
        Matrix with pattern summaries in rows and attributes they contain in columns. 
    """
    nb_p_s = len(pattern_summaries)
    pattern_summaries_attributes = np.zeros((nb_p_s, m))
    for i, p_s in enumerate(pattern_summaries):
        for attr in p_s[1]:
            pattern_summaries_attributes[i, attr] += 1
    
    return pattern_summaries_attributes

# TODO: Clean this function
def build_pattern_attributes(biadjacency, labels_louvain, kmeans_gnn_labels, 
                                kmeans_spectral_labels, kmeans_doc2vec_labels):
    """Build pattern x attributes matrix for all methods. """

    # patterns from Unexpectedness algorithm 
    #patterns_attributes = np.zeros((len(result), biadjacency.shape[1]))
    #for i, c in enumerate(result[1:]):
    #    patterns_attributes[i, c[1]] = 1
    
    # Pattern x attributes matrices for all methods
    #pattern_summarized_attributes = pattern_attributes(biadjacency, labels_cc_summarized)
    pattern_louvain_attributes = pattern_attributes(biadjacency, labels_louvain)
    pattern_gnn_kmeans_attributes = pattern_attributes(biadjacency, kmeans_gnn_labels)
    pattern_spectral_kmeans_attributes = pattern_attributes(biadjacency, kmeans_spectral_labels)
    pattern_doc2vec_kmeans_attributes = pattern_attributes(biadjacency, kmeans_doc2vec_labels)

    return pattern_louvain_attributes, pattern_gnn_kmeans_attributes, pattern_spectral_kmeans_attributes, pattern_doc2vec_kmeans_attributes

def density(adjacency: sparse.csr_matrix) -> float:
    """Density of directed graph. 
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Ajdacency matrix of the graph
    """
    # Remove self-nodes
    adjacency.setdiag(np.zeros(adjacency.shape[0]))
    adjacency.eliminate_zeros()

    m = adjacency.nnz
    n = adjacency.shape[0]
    
    if n == 1:
        return 0

    d = m / (n * (n - 1))

    return d

def kcore_decomposition(adjacency: sparse.csr_matrix) -> np.ndarray:
    """K-core decomposition algorithm.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph

    Returns
    -------
        Array of corresponding k-core for each node in the graph.
    """
    
    # Remove self-nodes
    adjacency.setdiag(np.zeros(adjacency.shape[0]))
    adjacency.eliminate_zeros()

    core = CoreDecomposition()
    cores_labels = core.fit_transform(directed2undirected(adjacency))

    return cores_labels

def smoothing(x: int, alpha: float=0.5, delta: int=10):
    """Smoothing function.

    Parameters
    ----------
    x : int
        Value to smoothen
    alpha : float, optional
        Smoothing parameter, by default 0.5
    delta : int, optional
        Smoothing parameter, by default 10

    Returns
    -------
    Float
        Smoothen value for x 
    """
    return (1 / (1 + np.exp(-alpha * (x - delta)))) 

def shuffle_columns(X, indexes):
    """Shuffle columns

    Parameters
    ----------
    X : 
        Either the biadjacency matrix of the attributed graph or an array of attributes.
    indexes : _type_
        Indexes of attributes

    Returns
    -------
    _type_
        Shuffled columns 
    """
    x = X.copy() 
    start = np.min(indexes)
    end = np.max(indexes) + 1

    if isinstance(X, sparse.csr_matrix):
        x[:, [np.arange(start, end)]] = X[:, indexes]
        x.eliminate_zeros()
    elif isinstance(X, np.ndarray):
        x[np.arange(start, end)] = X[indexes]    

    return x

def save_gensim_model(model, inpath, name):
    model.save(f"{inpath}/{name}.model")

def load_gensim_model(inpath, name):
    model = gensim.models.Doc2Vec.load(f'{inpath}/{name}.model')
    return model

def get_gensim_model(inpath: str, name: str, biadjacency: sparse.csr_matrix, names_col: np.ndarray):
    """Load gensim model if exists, otherwise train a new gensim model and save it.

    Parameters
    ----------
    inpath : str
        Path for existing model
    name : str
        Model name
    biadjacency : sparse.csr_matrix
        Biadjacency matrix of the graph
    names_col : np.ndarray
        Array of attribute names

    Returns
    -------
        Trained Gensim model
    """
    if not os.path.exists(f'{inpath}/{name}.model'):
        corpus = list(MyCorpus(biadjacency, names_col))
        model = gensim.models.doc2vec.Doc2Vec(vector_size=15, min_count=5, epochs=300)
        model.build_vocab(corpus)
        # Training model
        print('Training gensim model...')
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        # Save model
        save_gensim_model(model, inpath, name)
    else:
        model = load_gensim_model(inpath, name)
        print(f'Pre-trained gensim model loaded from {inpath}/{name}.model')

    return model
    
def get_root_directory():
    """Return root directory."""
    return os.path.dirname(os.path.realpath(__file__))
    