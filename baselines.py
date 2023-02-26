import os
import gensim
import numpy as np
from scipy import sparse

from corpus import MyCorpus

from sknetwork.clustering import Louvain, KMeans
from sknetwork.gnn import GNNClassifier
from sknetwork.utils import KMeansDense

from utils import load_gensim_model, save_gensim_model


# Louvain
resolutions = {'wikivitals-fr': {1: 0, 3: 0, 5: 0.3, 6: 0.4, 8: 0.6, 9: 0.7, 10: 0.8, 11: 0.9, 12: 0.9, 14: 1.05, 15: 1.1, 20: 1.4, 26: 1.86, 27: 1.89, 47: 3, 53: 3.8, 58: 3.87, 64: 4, 85: 5.58}, 
                'wikivitals': {4: 0.5, 7: 0.8, 11: 1, 13: 1.3, 14: 1.18, 16: 1.5, 18: 1.6, 19: 1.6, 21: 1.76, 23: 1.83, 28: 2.3, 29: 2.4, 31: 2.5, 34: 2.7, 39: 3.1, 40: 3.2, 47: 3.903, 49: 3.95, 57: 4.2, 64: 4.6, 65: 4.70, 85: 5.58, 101: 6.4, 109: 6.7, 118: 7.25}, 
                'wikischools': {3: 0.4, 4: 0.45, 8: 1, 3: 0.4, 8: 0.58, 9: 0.9, 10: 1.18, 13: 1.4, 15: 1.5, 18: 1.69, 22: 1.95, 24: 2.1, 25: 2.2, 29: 2.3, 32: 2.51, 37: 2.6, 39: 2.75, 40: 2.75,  46: 2.99, 49: 3.1, 52: 3.3, 57: 3.7, 58: 3.7, 70: 4.2, 71: 4.4, 73: 4.41, 74: 4.45, 79: 4.59},
                'lastfm': {60: 2.74, 32: 0.65},
                'sanFranciscoCrimes': {5: 0.13, 6: 0.14},
                'london': {1: 0.05}}


def get_louvain(dataset: str, adjacency: sparse.csr_matrix, nb_cc: int) -> np.ndarray:
    """Louvain algorithm for clustering graphs by maximization of modularity. Returns labels of the nodes.
    
    Parameters
    ----------
    dataset: str
        Name of dataset on netset.
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    nb_cc: int
        Number of communities
    
    Outputs
    -------
        Array of node labels.
    """
    louvain = Louvain(resolution=resolutions.get(dataset).get(nb_cc)) 
    labels_louvain = louvain.fit_transform(adjacency)
    nb_louvain = len(np.unique(labels_louvain))
    
    return labels_louvain

def get_gnn(adjacency: sparse.csr_matrix, biadjacency: sparse.csr_matrix, labels: np.ndarray, hidden_dim: int, nb_cc: int) -> np.ndarray:
    """GNN embedding + KMeans clustering. Returns labels of the nodes.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    biadjacency: sparse.csr_matrix
        Biadjacency matrix of the graph
    labels: np.ndarray
        Node labels
    hidden_dim: int
        Hidden layer dimension
    nb_cc: int
        Number of communities (for KMeans clustering)
    
    Outputs
    -------
        Array of node labels.
    """
    features = biadjacency
    n_labels = len(np.unique(labels))
    gnn = GNNClassifier(dims=[hidden_dim, n_labels],
                        layer_types='conv',
                        activations=['Relu', 'Softmax'],
                        verbose=False)

    # Train GNN model
    gnn.fit(adjacency, features, labels, train_size=0.8, val_size=0.1, test_size=0.1, n_epochs=50)
    
    # KMeans on GNN node embedding
    gnn_embedding = gnn.layers[-1].embedding
    kmeans = KMeansDense(n_clusters=nb_cc) # k = number of connected components in summarized graph
    kmeans_gnn_labels = kmeans.fit_transform(gnn_embedding)

    return kmeans_gnn_labels

def get_spectral(adjacency: sparse.csr_matrix,  nb_cc: int) -> np.ndarray:
    """Spectral embedding + KMeans clustering. Returns labels of the nodes.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    nb_cc: int
        Number of communities (for KMeans clustering)
    
    Outputs
    -------
        Array of node labels.
    """
    # Spectral + KMeans
    kmeans = KMeans(n_clusters=nb_cc) # k = number of connected components in summarized graph
    kmeans_spectral_labels = kmeans.fit_transform(adjacency)

    return kmeans_spectral_labels

def get_doc2vec(biadjacency: sparse.csr_matrix, d: str, names_col: np.ndarray, nb_cc: int)->np.ndarray:
    """Doc2Vec embedding + KMeans clustering. Returns labels of the nodes.
    
    Parameters
    ----------
    biadjacency: sparse.csr_matrix
        Biadjacency matrix of the graph
    names_col: np.ndarray
        Array of feature names.
    nb_cc: int
        Number of communities (for KMeans clustering)
    
    Outputs
    -------
        Array of node labels.
    """
    if not os.path.exists(f'models/gensim_model_{d}.model'):
        # Build corpus
        corpus = list(MyCorpus(biadjacency, names_col))
        # Build model
        model = gensim.models.doc2vec.Doc2Vec(vector_size=15, min_count=5, epochs=300)
        model.build_vocab(corpus)
        # Train model
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        # Save model
        save_gensim_model(model, 'models/', f'gensim_model_{d}')
    else:
        model = load_gensim_model('models/', f'gensim_model_{d}')

    # Kmeans on embeddings
    kmeans = KMeansDense(n_clusters=nb_cc) # k = number of connected components in summarized graph
    kmeans_doc2vec_labels = kmeans.fit_transform(model.dv.vectors)

    return kmeans_doc2vec_labels

def get_community_graph(adjacency: sparse.csr_matrix, labels_communities: np.ndarray) -> sparse.csr_matrix:
    """Equivalent of summarized graph but for community-based methods. Returns the adjacency matrix of the graph made of the union of all communities. 
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    labels_communities: np.ndarray
        Array of node community labels 

    Output
    ------
        Sparse matrix of the community graph.
    """
    n_com = len(np.unique(labels_communities))
    rows, cols = [], []
    for n in range(n_com):
        nodes = np.flatnonzero(labels_communities == n)
        idx = 0
        idx_nodes = np.array([-1] * len(nodes)) # number of unique nodes from communities
        # reindex nodes
        for n in nodes:
            if n not in idx_nodes:
                idx_nodes[idx] = n
                idx += 1

        # Record edges from subgraph related to community
        adj_com = adjacency[nodes, :][:, nodes].tocoo()
        reindex_rows = [int(idx_nodes[src]) for src in adj_com.row]
        reindex_cols = [int(idx_nodes[dst]) for dst in adj_com.col]
        rows += reindex_rows
        cols += reindex_cols
        
    return sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=adjacency.shape).tocsr()