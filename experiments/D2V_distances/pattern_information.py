from collections import defaultdict
import gensim
from gensim.models import Doc2Vec
import numpy as np
import os
import pickle
from scipy import sparse
from scipy.stats import wasserstein_distance
import tempfile
from tqdm import tqdm

from sknetwork.data import load_netset
from sknetwork.utils import get_degrees
from sknetwork.topology import get_connected_components


def save_gensim_model(model, inpath, name):
    model.save(f"{inpath}/{name}.model")

def load_gensim_model(inpath, name):
    model = Doc2Vec.load(f'{inpath}/{name}.model')
    return model

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
        with open(f'../data/{dataset}Graph', 'br') as f:
            graph = pickle.load(f)

    adjacency = graph.adjacency
    biadjacency = graph.biadjacency
    names = graph.names
    names_col = graph.names_col
    
    return adjacency, biadjacency, names, names_col, labels

class MyCorpus():
    """A memory-friendly iterator that yields documents as TaggedDocument objects, i.e tokens associated with index of document."""
    
    def __init__(self, data, vocab, tokens_only=False):
        self.data = data
        self.vocab = vocab
        self.tokens_only = tokens_only
    
    def __iter__(self):
        if isinstance(self.data, sparse.csr_matrix):
            for i, x in enumerate(self.data):
                tokens = list(self.vocab[x.indices])
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
        else:
            if not self.tokens_only:
                for i, x in enumerate(self.data):
                    tokens = list(self.vocab[np.flatnonzero(x)])
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
            else:
                for i, x in enumerate(self.data):
                    tokens = list(self.vocab[np.flatnonzero(x)])
                    yield tokens
                    
def preprocess_data(biadjacency, names_col, s, sort_data=True):
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

def d2v_embedding(model, doc):
    """Use pre-trained model to embed document."""
    return model.infer_vector(doc)

def pairwise_wd_distance(matrix, n, model, names):
    """Doc2Vec embedding + pairwise Wasserstein distances between elements in matrix."""

    wd_matrix = np.zeros((n, n))
    
    for i in tqdm(range(n)):
        w1 = d2v_embedding(model, names[np.flatnonzero(matrix[i, :])])
        for j in range(n):
            w2 = d2v_embedding(model, names[np.flatnonzero(matrix[j, :])])
            wd_matrix[i, j] = wasserstein_distance(w1, w2)
            
    return wd_matrix

def diversity(pw_distances: np.ndarray, delta: float=0.2) -> float:
    """Diversity, i.e. ratio between number of pairwise distances above threshold and total number of distances. 
    
    Parameters
    ----------
    pw_distances: np.ndarray
        Pairwise distances.
    delta: float (default=0.2)
        Minimumm pairwise distance threshold.
        
    Outputs
    -------
        Diversity. 
    """
    n = pw_distances.shape[0]
    upper = pw_distances[np.triu_indices(n)]
    nb_ps = np.sum(upper > delta)
    
    return (nb_ps / len(upper)).item()

def coverage(patterns, n):
    all_nodes = set()
    for p in patterns:
        if len(p[1]) > 0:
            all_nodes |= set(p[0])

    cov = len(all_nodes) / n
    return cov

def conciseness(patterns):
    n_nodes, n_attrs = [], []
    for p in patterns:
        if len(p[1])>0:
            nodes_idxs = set(p[0])
            attr_idxs = set(p[1])
            n_nodes.append(len(nodes_idxs))
            n_attrs.append(len(attr_idxs))

    conc = np.mean(n_nodes) + np.mean(n_attrs)
    print( len(patterns), np.mean(n_nodes) , np.mean(n_attrs))
    return conc.item() * len(patterns)

def conciseness_summaries(adjacency, biadjacency, nb_cc, labels_cc_summarized, concept_summarized_attributes) -> float:
    """Conciseness of summarized graph, i.e. sum of the median of number of nodes and median of number of attributes per pattern/community.
    
    Outputs
    -------
        Conciseness. 
    """
    extent_size_summaries, intent_size_summaries = [], []
    print(concept_summarized_attributes.shape)
    for i in range(nb_cc):
        mask_cc = labels_cc_summarized == i
        extent_size_summaries.append(mask_cc.sum())
        intent_size_summaries.append(len(np.flatnonzero(concept_summarized_attributes[i, :])))

    avg_deg = np.mean(get_degrees(adjacency.astype(bool)))
    avg_deg_attr = np.mean(get_degrees(biadjacency.astype(bool)))
    print(f'Avg subgraph size: {np.mean(extent_size_summaries)} - avg node degree: {avg_deg}')
    print(f'Avg subgraph attributes: {np.mean(intent_size_summaries)} - avg nb attr: {avg_deg_attr}')
    return nb_cc * ((np.mean(extent_size_summaries) ) * (np.mean(intent_size_summaries) ))

def get_summarized_graph(adjacency: sparse.csr_matrix, patterns: list) -> sparse.csr_matrix:
    """Get summarized graph given patterns and original adjacency matrix. 
       A summarized graph is union of all subgraphs from a list of patterns.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    patterns: list  
        List of tuples where each tuple is an unexpected pattern made of (extent, intent).  

    Returns
    -------
        CSR matrix of the summarized graph.       
    """
    
    rows, cols = [], []

    for c in patterns:

        # exclude first element of lattice 
        if len(c[1]) > 0:
            nodes = sorted(c[0])
            idx = 0
            idx_nodes = np.array([-1] * len(nodes)) # number of unique nodes from patterns
            # reindex nodes
            for n in nodes:
                if n not in idx_nodes:
                    idx_nodes[idx] = n
                    idx += 1
            
            # Record edges from subgraph related to pattern
            adj_pattern = adjacency[nodes, :][:, nodes].tocoo()
            reindex_rows = [int(idx_nodes[src]) for src in adj_pattern.row]
            reindex_cols = [int(idx_nodes[dst]) for dst in adj_pattern.col]
            rows += reindex_rows
            cols += reindex_cols
            
    return sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=adjacency.shape).tocsr()

def get_summarized_biadjacency(adjacency: sparse.csr_matrix, biadjacency: sparse.csr_matrix, patterns: list) -> sparse.csr_matrix:
    """Get summarized biadjacency matrix given an original graph and a list of patterns. Summarized biadjacency contains all links between nodes and attributes that are induced by a summarized graph.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    biadjacency: sparse.csr_matrix
        Biadjacency matrix of the graph
    patterns:  list
        List of tuples where each tuple is an unexpected pattern made of (extent, intent).  

    Returns
    -------
        CSR matrix of the summarized biadjacency matrix.   
    """
    summarized_biadj = np.zeros((adjacency.shape[0], biadjacency.shape[1]))
    for p in patterns:
        if len(p[1]) > 0:
            for node in p[0]:
                summarized_biadj[node, p[1]] += 1

    summarized_biadj = sparse.csr_matrix(summarized_biadj.astype(bool), shape=summarized_biadj.shape)
    
    return summarized_biadj

def get_pattern_summaries(summarized_adjacency: sparse.csr_matrix):
    """Extract connected components from a summarized graph and return labels. Labels are returned only for nodes in a connected component with size > 1.
    
    Parameters
    ----------
    summarized_adjacency: sparse.csr_matrix
        Adjacency matrix of the summarized graph.
        
    Outputs
    -------
        Array of labels, node mask. """
    # Summarized graph filtered on used nodes
    mask = np.flatnonzero(summarized_adjacency.dot(np.ones(summarized_adjacency.shape[1])))
    
    # Number of connected components (NOT considering isolated nodes)
    labels_cc_summarized = get_connected_components(summarized_adjacency[mask, :][:, mask])
    
    return labels_cc_summarized, mask   

def pattern_attributes(biadjacency, labels):
    """Build pattern x attributes matrix. Column values are count of occurrences of attributes for each pattern/community.
    
    Parameters
    ----------
    biadjacency: sparse.csr_matrix
        Biadjacency matrix of the graph
    labels: np.ndarray
        Belonging community for each node in the graph, e.g Louvain labels or KMeans labels
        
    Outputs
    -------
        Matrix with patterns/communities in rows and count of attributes in columns. """

    nb_cc = len(np.unique(labels))
    matrix = np.zeros((nb_cc, biadjacency.shape[1]))
    for c in range(nb_cc):
        mask_cc = labels == c
        indices_attr = biadjacency[mask_cc].indices
        for ind in indices_attr:
            matrix[c, ind] += 1

    return matrix

def coverage_summarized(summarized_adjacency: sparse.csr_matrix) -> float:
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


if __name__=='__main__':

    datasets = ['sanFranciscoCrimes']
    betas = [5]
    ss = [8, 7, 6, 5]
    deltas = [0]
    method = 'summaries'
    print(f'Method: {method}')

    informations_dict = defaultdict(list)

    for d in datasets:
        print(f'**Dataset: {d}')
        informations_dict[d] = defaultdict(dict)

        # Load netset data
        adjacency, biadjacency, names, names_col, labels = load_data(d)
        
        # Gensim model
        if not os.path.exists(f'models/gensim_model_{d}.model'):
            corpus = list(MyCorpus(biadjacency, names_col))
            model = gensim.models.doc2vec.Doc2Vec(vector_size=15, min_count=5, epochs=300)
            model.build_vocab(corpus)
            # Training model
            print('Training gensim model...')
            model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
            # Save model
            save_gensim_model(model, 'models/', f'gensim_model_{d}')
        else:
            model = load_gensim_model('models/', f'gensim_model_{d}')
            print('Pre-trained model loaded')

        for b in betas:
            informations_dict[d][b] = defaultdict(dict)
            for s in ss:
                informations_dict[d][b][s] = defaultdict(dict)
                for delt in deltas:
                    print(f' -delta: {delt}')

                    # Load patterns
                    #with open(f'../output/result/delta_{delt}/result_{d}_{b}_{s}_orderTrue_delta_{delt}.bin', "rb") as data:
                    #    patterns = pickle.load(data)
                    data_path = f'/Users/simondelarue/Documents/PhD/Research/Co-Meg/CoMEG/output/result/with_prob'
                    if d in ['wikivitals', 'wikivitals-fr', 'wikischools']:
                        with open(f'{data_path}/result_{d}_{b}_{s}_orderTrue_delta_{delt}.bin', "rb") as data:
                            patterns = pickle.load(data)
                    else:
                        with open(f'{data_path}/result_{d}_{b}_{s}_orderTrue_delta_{delt}.bin', "rb") as data:
                            patterns = pickle.load(data)
                    print(f'Number of patterns: {len(patterns)}')

                    # Preprocess data (get same attribute order as in UnexPattern)
                    new_biadjacency, words = preprocess_data(biadjacency, names_col, s, sort_data=False)
                    print(f'  Filtered biadjacency: {new_biadjacency.shape}')

                    # List of all patterns
                    nb_patterns = len(patterns)

                    # Pattern x attributes matrix
                    patterns_attributes = np.zeros((nb_patterns, new_biadjacency.shape[1]))
                    for i, p in enumerate(patterns):
                        patterns_attributes[i, p[1]] = 1
                    
                    # Summarized graph + features
                    summarized_adj = get_summarized_graph(adjacency, patterns)
                    summarized_biadj = get_summarized_biadjacency(adjacency, new_biadjacency, patterns)
                    
                    # Pattern summaries 
                    pattern_summary_labels, mask = get_pattern_summaries(summarized_adj)
                    n_p_summaries = len(np.unique(pattern_summary_labels))
                    pattern_summarized_attributes = pattern_attributes(new_biadjacency, pattern_summary_labels)

                    # Wasserstein distances
                    #inpath_w = f'/Users/simondelarue/Documents/PhD/Research/Co-Meg/CoMEG/output/result/delta_{delt}'
                    inpath_w = f'/Users/simondelarue/Documents/PhD/Research/Co-Meg/CoMEG/output/result/with_prob'

                    w_filename = f'wasserstein_distances_{d}_{b}_{s}_delta_{delt}_{method}.pkl'

                    if not os.path.exists(f'{inpath_w}/{w_filename}'):
                        print(f'  Computing Wasserstein distances...')
                        if method == "summaries":
                            d2v_wd_matrix = pairwise_wd_distance(pattern_summarized_attributes, n_p_summaries, model, words)
                        elif method == 'patterns':
                            d2v_wd_matrix = pairwise_wd_distance(patterns_attributes, nb_patterns, model, words)
                        # Save distances
                        with open(f'{inpath_w}/{w_filename}', 'wb') as f:
                            np.save(f, d2v_wd_matrix)
                    else:
                        print(f'  Using pre-computed Wasserstein distances')
                        with open(f'{inpath_w}/{w_filename}', 'rb') as data:
                            d2v_wd_matrix = np.load(data)
                
                    # Information
                    print(f'Computing information metrics...')
                    div = diversity(d2v_wd_matrix, delta=0.01)
                    #div = diversity(d2v_wd_matrix, delta=0.2)
                    if method == 'patterns':
                        cov = coverage(patterns, adjacency.shape[0])
                        conc = conciseness(patterns)
                        print(conc)
                    elif method == 'summaries':
                        cov = coverage_summarized(summarized_adj)
                        conc = conciseness_summaries(adjacency, biadjacency, n_p_summaries, pattern_summary_labels, pattern_summarized_attributes)
                    information = ((div * cov) /conc)
                    
                    #with open(f'{inpath_w}/information_details_{d}_{b}_{s}_delta_{delt}_summaries.txt', 'w') as f:
                    #    f.write(f'{div}, {cov}, {conc}, {information}')
                    with open(f'{inpath_w}/information_details_{d}_{b}_{s}_delta_{delt}_{method}.txt', 'w') as f:
                        f.write(f'{div}, {cov}, {conc}, {information}')
                    print(f'  Done! Results saved in {inpath_w}')