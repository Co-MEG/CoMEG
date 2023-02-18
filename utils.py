import gensim
import os
import numpy as np
from scipy import sparse

from sknetwork.topology import CoreDecomposition
from sknetwork.utils import directed2undirected

from experiments.pattern_information import MyCorpus

def pattern_attributes(biadjacency, labels, mask=None):
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

def build_pattern_attributes(result, biadjacency, labels_cc_summarized, labels_louvain, kmeans_gnn_labels, 
                                kmeans_spectral_labels, kmeans_doc2vec_labels):
    """Build pattern x attributes matrix for all methods. """

    # patterns from Unexpectedness algorithm 
    patterns_attributes = np.zeros((len(result), biadjacency.shape[1]))
    for i, c in enumerate(result[1:]):
        patterns_attributes[i, c[1]] = 1
    
    # Pattern x attributes matrices for all methods
    pattern_summarized_attributes = pattern_attributes(biadjacency, labels_cc_summarized)
    pattern_louvain_attributes = pattern_attributes(biadjacency, labels_louvain)
    pattern_gnn_kmeans_attributes = pattern_attributes(biadjacency, kmeans_gnn_labels)
    pattern_spectral_kmeans_attributes = pattern_attributes(biadjacency, kmeans_spectral_labels)
    pattern_doc2vec_kmeans_attributes = pattern_attributes(biadjacency, kmeans_doc2vec_labels)

    return patterns_attributes, pattern_summarized_attributes, pattern_louvain_attributes, pattern_gnn_kmeans_attributes, pattern_spectral_kmeans_attributes, pattern_doc2vec_kmeans_attributes

def density(g: sparse.csr_matrix) -> float:
    """Density of directed graph. 
    
    Parameters
    ----------
    g: sparse.csr_matrix
        Graph
    """
    # Remove self-nodes
    g.setdiag(np.zeros(g.shape[0]))
    g.eliminate_zeros()

    m = g.nnz
    n = g.shape[0]
    
    if n == 1:
        return 0

    d = m / (n * (n - 1))

    return d

def kcore_decomposition(g: sparse.csr_matrix) -> np.ndarray:
    """K-core decomposition algorithm.
    
    Parameters
    ----------
    g: sparse.csr_matrix
        Graph

    Returns
    -------
        Array of corresponding k-core for each node in the graph.
    """
    
    # Remove self-nodes
    g.setdiag(np.zeros(g.shape[0]))
    g.eliminate_zeros()

    core = CoreDecomposition()
    cores_labels = core.fit_transform(directed2undirected(g))

    return cores_labels

def smoothing(x, alpha=0.5, delta=10):
    return (1 / (1 + np.exp(-alpha * (x - delta)))) 

def shuffle_columns(X, indexes):
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

def get_gensim_model(inpath, name, biadjacency, names_col):

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
    