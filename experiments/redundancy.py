# Analysis of pattern redundancy (RQ2)

# **************************************************************************
#   1. Consider each node as a document containing words, and train a document 
#      embedding model using `Doc2Vec`  
#   2. Use trained model to infer embeddings for new documents, 
#      i.e patterns/concepts/communities  
#   3. Compute Wasserstein distances between these embeddings
# **************************************************************************

from collections import defaultdict
import gensim
import numpy as np
import pickle
from scipy import sparse

from sknetwork.clustering import Louvain, KMeans
from sknetwork.data import load_netset
from sknetwork.gnn import GNNClassifier
from sknetwork.topology import get_connected_components
from sknetwork.utils import get_degrees, KMeansDense

from corpus import MyCorpus
from distances import pairwise_wd_distance
from summarization import get_summarized_graph
from utils import build_pattern_attributes


def load_result(path, filename):
    with open(f'{path}/{filename}.bin', 'rb') as data:
        result = pickle.load(data)
    return result

datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']
sorted_attributes = True
ss = [8, 7, 6, 5]
betas = [8, 7, 6, 5]
avgs = defaultdict(dict)
outpath = 'output/result'
resolutions = {'wikivitals-fr': {1: 0, 3: 0, 5: 0.3, 6: 0.4, 10: 0.8, 12: 0.9, 15: 1.1, 47: 3, 27: 1.89}, 
                'wikivitals': {4: 0.5, 7: 0.8, 11: 1, 13: 1.3, 14: 1.18, 16: 1.5, 19: 1.6, 21: 1.76, 29: 2.4, 31: 2.5, 34: 2.7, 47: 3.903, 49: 3.95, 65: 4.70, 101: 6.4}, 
                'wikischools': {4: 0.45, 8: 1, 3: 0.4, 8: 0.58, 9: 0.9, 10: 1.18, 13: 1.4, 15: 1.5, 18: 1.69, 22: 1.95, 39: 2.75, 40: 2.75,  46: 2.99, 58: 3.7, 73: 4.41}}

# Run experiment
# ------------------------------------------------------------------
for dataset in datasets:
    avgs[dataset] = defaultdict(dict)
    for b in betas:
        avgs[dataset][b] = defaultdict(dict)
        for s in ss:
            avgs[dataset][b][s] = defaultdict(list)

            print(f'* Dataset: {dataset} - beta={b} - s={s}')
            
            # Load result
            # ------------------------------------------------------------------
            filename = f'result_{dataset}_{b}_{s}_order{str(sorted_attributes)}'
            result = load_result(outpath, filename)
            
            # Load and preprocess data
            graph = load_netset(dataset)
            adjacency = graph.adjacency
            biadjacency = graph.biadjacency
            names = graph.names
            words = graph.names_col
            labels = graph.labels
            names_labels = graph.names_labels
            orig_words = words.copy()
            
            # Degree of attribute = # articles in which it appears
            freq_attribute = get_degrees(biadjacency.astype(bool), transpose=True)
            index = np.flatnonzero((freq_attribute <= 1000) & (freq_attribute >= s))

            # Filter data with index
            biadjacency = biadjacency[:, index]
            words = words[index]
            freq_attribute = freq_attribute[index]

            # Order attributes according to their ascending degree
            # This allows to add first attributes that will generate bigger subgraphs
            if sorted_attributes:
                sort_index = np.argsort(freq_attribute)
            else:
                sort_index = np.arange(0, len(freq_attribute))
            sorted_degs = freq_attribute[sort_index]
            filt_biadjacency = biadjacency[:, sort_index]
            sorted_names_col = words[sort_index]

            # Graph summarization
            # ------------------------------------------------------------------
            summarized_adjacency = get_summarized_graph(adjacency, result)

            # Summarized graph filtered on used nodes
            mask = np.flatnonzero(summarized_adjacency.dot(np.ones(summarized_adjacency.shape[1])))

            # Summarized biadjacency
            summarized_biadjacency = np.zeros((adjacency.shape[0], biadjacency.shape[1]))
            for c in result:
                if len(c[1]) > 0:
                    for node in c[0]:
                        summarized_biadjacency[node, c[1]] += 1
            summarized_biadjacency = sparse.csr_matrix(summarized_biadjacency.astype(bool), shape=summarized_biadjacency.shape)
            
            # Number of connected components NOT considering isolated nodes
            labels_cc_summarized = get_connected_components(summarized_adjacency[mask, :][:, mask])
            nb_cc = len(np.unique(labels_cc_summarized)) 
            print(nb_cc)

            # Baseline methods
            # ------------------------------------------------------------------

            # Louvain
            louvain = Louvain(resolution=resolutions.get(dataset).get(nb_cc)) 
            labels_louvain = louvain.fit_transform(adjacency)
            nb_louvain = len(np.unique(labels_louvain))

            # GNN
            features = biadjacency
            hidden_dim = 16
            n_labels = len(np.unique(labels))
            gnn = GNNClassifier(dims=[hidden_dim, n_labels],
                                layer_types='conv',
                                activations=['Relu', 'Softmax'],
                                verbose=False)

            gnn.fit(adjacency, features, labels, train_size=0.8, val_size=0.1, test_size=0.1, n_epochs=50)
            # KMeans on GNN node embedding
            gnn_embedding = gnn.layers[-1].embedding
            kmeans = KMeansDense(n_clusters=nb_cc) # k = number of connected components in summarized graph
            kmeans_gnn_labels = kmeans.fit_transform(gnn_embedding)

            # Spectral + KMeans
            kmeans = KMeans(n_clusters=nb_cc) # k = number of connected components in summarized graph
            kmeans_spectral_labels = kmeans.fit_transform(adjacency)
            
            # Doc2Vec model on whole biadjacency matrix
            corpus = list(MyCorpus(biadjacency, orig_words))
            model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=10, epochs=50)
            model.build_vocab(corpus)
            # Training model
            model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

            # KMeans model on d2v embeddings
            kmeans = KMeansDense(n_clusters=nb_cc) # k = number of connected components in summarized graph
            kmeans_doc2vec_labels = kmeans.fit_transform(model.dv.vectors)
            
            # Concept x attributes matrices for each method
            # ------------------------------------------------------------------
            concept_attributes, concept_summarized_attributes, concept_louvain_attributes, concept_gnn_kmeans_attributes, concept_spectral_kmeans_attributes, concept_doc2vec_kmeans_attributes = build_pattern_attributes(result, 
                                                                                                    biadjacency, labels_cc_summarized, labels_louvain, kmeans_gnn_labels, kmeans_spectral_labels, kmeans_doc2vec_labels)
            
            # Pairwise Wassertein distances between embeddings
            # ------------------------------------------------------------------
            d2v_wd_matrix_concepts = pairwise_wd_distance(concept_attributes, nb_cc, model, sorted_names_col)
            d2v_wd_matrix_summarized = pairwise_wd_distance(concept_summarized_attributes, nb_cc, model, sorted_names_col)
            d2v_wd_matrix_louvain = pairwise_wd_distance(concept_louvain_attributes, nb_louvain, model, words)
            d2v_wd_matrix_gnn_kmeans = pairwise_wd_distance(concept_gnn_kmeans_attributes, nb_cc, model, words)
            d2v_wd_matrix_spectral_kmeans = pairwise_wd_distance(concept_spectral_kmeans_attributes, nb_cc, model, words)
            d2v_wd_matrix_d2v_kmeans = pairwise_wd_distance(concept_doc2vec_kmeans_attributes, nb_cc, model, words)
            
            # Save mean values in dictionary
            avgs[dataset][b][s]['concepts'].append(np.mean(d2v_wd_matrix_concepts))
            avgs[dataset][b][s]['summarized'].append(np.mean(d2v_wd_matrix_summarized))
            avgs[dataset][b][s]['louvain'].append(np.mean(d2v_wd_matrix_louvain))
            avgs[dataset][b][s]['spectral_kmeans'].append(np.mean(d2v_wd_matrix_spectral_kmeans))
            avgs[dataset][b][s]['gnn_kmeans'].append(np.mean(d2v_wd_matrix_gnn_kmeans))
            avgs[dataset][b][s]['doc2vec_kmeans'].append(np.mean(d2v_wd_matrix_d2v_kmeans))

# Save result
with open(f'output/result/average_wasserstein_distances_all.pkl', 'wb') as f:
    pickle.dump(avgs, f)
