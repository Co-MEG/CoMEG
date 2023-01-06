# Analysis of pattern vs baselines (RQ2)

# **************************************************************************
#   1. Compute statistics upon patterns/communities to compare them, such as:
#           - average size of graph, size of attribute set
#           - average density
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
from summarization import get_summarized_graph
from utils import build_pattern_attributes, density


def load_result(path, filename):
    with open(f'{path}/{filename}.bin', 'rb') as data:
        result = pickle.load(data)
    return result

# Parameters 
# ====================================================================
datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']
sorted_attributes = True
ss = [8, 7, 6, 5]
betas = [5]
extent_sizes_dict = defaultdict(dict)
intent_sizes_dict = defaultdict(dict)
densities_dict = defaultdict(dict)
outpath = 'output/result'
resolutions = {'wikivitals-fr': {1: 0, 3: 0, 5: 0.3, 6: 0.4, 10: 0.8, 12: 0.9, 15: 1.1, 47: 3, 27: 1.89}, 
                'wikivitals': {4: 0.5, 7: 0.8, 11: 1, 13: 1.3, 14: 1.18, 16: 1.5, 19: 1.6, 21: 1.76, 29: 2.4, 31: 2.5, 34: 2.7, 47: 3.903, 49: 3.95, 65: 4.70, 101: 6.4}, 
                'wikischools': {4: 0.45, 8: 1, 3: 0.4, 8: 0.58, 9: 0.9, 10: 1.18, 13: 1.4, 15: 1.5, 18: 1.69, 22: 1.95, 39: 2.75, 40: 2.75,  46: 2.99, 58: 3.7, 73: 4.41}}

extent_size = False
intent_size = False
densities = True
# ====================================================================


# Run experiment
# ------------------------------------------------------------------
for dataset in datasets:
    extent_sizes_dict[dataset] = defaultdict(dict)
    intent_sizes_dict[dataset] = defaultdict(dict)
    densities_dict[dataset] = defaultdict(dict)
    for b in betas:
        extent_sizes_dict[dataset][b] = defaultdict(dict)
        intent_sizes_dict[dataset][b] = defaultdict(dict)
        densities_dict[dataset][b] = defaultdict(dict)
        for s in ss:
            extent_sizes_dict[dataset][b][s] = defaultdict(dict)
            intent_sizes_dict[dataset][b][s] = defaultdict(dict)
            densities_dict[dataset][b][s] = defaultdict(dict)

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
            
            # Statistics on patterns/communities
            # ------------------------------------------------------------------
            
            # Size of extents/intents
            if extent_size:
                extent_size_patterns = [len(p[0]) for p in result if len(p[1]) > 0]

                extent_size_summaries = []
                for i in range(nb_cc):
                    mask_cc = labels_cc_summarized == i
                    extent_size_summaries.append(mask_cc.sum())

                extent_size_louvain = []
                for i in range(nb_louvain):
                    mask_cc = labels_louvain == i
                    extent_size_louvain.append(mask_cc.sum())

                extent_size_gnn_kmeans = []
                for i in range(nb_cc):
                    mask_cc = kmeans_gnn_labels == i
                    extent_size_gnn_kmeans.append(mask_cc.sum())

                extent_size_kmeans_spectral = []
                for i in range(nb_cc):
                    mask_cc = kmeans_spectral_labels == i
                    extent_size_kmeans_spectral.append(mask_cc.sum())

                extent_size_d2v = []
                for i in range(nb_cc):
                    mask_cc = kmeans_doc2vec_labels == i
                    extent_size_d2v.append(mask_cc.sum())
                
                extent_sizes_dict[dataset][b][s]['patterns'] = extent_size_patterns
                extent_sizes_dict[dataset][b][s]['summaries'] = extent_size_summaries
                extent_sizes_dict[dataset][b][s]['louvain'] = extent_size_louvain
                extent_sizes_dict[dataset][b][s]['gnn'] = extent_size_gnn_kmeans
                extent_sizes_dict[dataset][b][s]['spectral'] = extent_size_kmeans_spectral
                extent_sizes_dict[dataset][b][s]['doc2vec'] = extent_size_d2v
            
            if intent_size:
                intent_size_patterns = [len(p[1]) for p in result if len(p[1]) > 0]

                intent_size_summaries = []
                for i in range(nb_cc):
                    intent_size_summaries.append(len(np.flatnonzero(concept_summarized_attributes[i, :])))

                intent_size_louvain = []
                for i in range(nb_louvain):
                    intent_size_louvain.append(len(np.flatnonzero(concept_louvain_attributes[i, :])))

                intent_size_gnn_kmeans = []
                for i in range(nb_cc):
                    intent_size_gnn_kmeans.append(len(np.flatnonzero(concept_gnn_kmeans_attributes[i, :])))

                intent_size_kmeans_spectral = []
                for i in range(nb_cc):
                    intent_size_kmeans_spectral.append(len(np.flatnonzero(concept_spectral_kmeans_attributes[i, :])))

                intent_size_d2v = []
                for i in range(nb_cc):
                    intent_size_d2v.append(len(np.flatnonzero(concept_doc2vec_kmeans_attributes[i, :])))
                
                intent_sizes_dict[dataset][b][s]['patterns'] = intent_size_patterns
                intent_sizes_dict[dataset][b][s]['summaries'] = intent_size_summaries
                intent_sizes_dict[dataset][b][s]['louvain'] = intent_size_louvain
                intent_sizes_dict[dataset][b][s]['gnn'] = intent_size_gnn_kmeans
                intent_sizes_dict[dataset][b][s]['spectral'] = intent_size_kmeans_spectral
                intent_sizes_dict[dataset][b][s]['doc2vec'] = intent_size_d2v

            # Graph Densities
            if densities:
                densities_patterns = []
                for p in result:
                    if len(p[1]) > 0:
                        subgraph = adjacency[p[0], :][:, p[0]]
                        densities_patterns.append(density(subgraph))

                densities_summaries = []
                for i in range(nb_cc):
                    mask_cc = labels_cc_summarized == i
                    subgraph = summarized_adjacency[mask, :][:, mask][mask_cc, :][:, mask_cc]
                    densities_summaries.append(density(subgraph))

                densities_louvain = []
                for i in range(nb_louvain):
                    mask_cc = labels_louvain == i
                    subgraph = summarized_adjacency[mask_cc, :][:, mask_cc]
                    densities_louvain.append(density(subgraph))

                densities_gnn_kmeans = []
                for i in range(nb_cc):
                    mask_cc = kmeans_gnn_labels == i
                    subgraph = summarized_adjacency[mask_cc, :][:, mask_cc]
                    densities_gnn_kmeans.append(density(subgraph))

                densities_kmeans_spectral = []
                for i in range(nb_cc):
                    mask_cc = kmeans_spectral_labels == i
                    subgraph = summarized_adjacency[mask_cc, :][:, mask_cc]
                    densities_kmeans_spectral.append(density(subgraph))

                densities_d2v = []
                for i in range(nb_cc):
                    mask_cc = kmeans_doc2vec_labels == i
                    subgraph = summarized_adjacency[mask_cc, :][:, mask_cc]
                    densities_d2v.append(density(subgraph))
                
                densities_dict[dataset][b][s]['patterns'] = densities_patterns
                densities_dict[dataset][b][s]['summaries'] = densities_summaries
                densities_dict[dataset][b][s]['louvain'] = densities_louvain
                densities_dict[dataset][b][s]['gnn'] = densities_gnn_kmeans
                densities_dict[dataset][b][s]['spectral'] = densities_kmeans_spectral
                densities_dict[dataset][b][s]['doc2vec'] = densities_d2v

# Save result
if extent_size:
    with open(f'output/result/extent_sizes.pkl', 'wb') as f:
        pickle.dump(extent_sizes_dict, f)
if intent_size:
    with open(f'output/result/intent_sizes.pkl', 'wb') as f:
        pickle.dump(intent_sizes_dict, f)
if densities:
    with open(f'output/result/densities.pkl', 'wb') as f:
        pickle.dump(densities_dict, f)