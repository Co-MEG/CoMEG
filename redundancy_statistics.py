from collections import defaultdict
import os
import numpy as np
from scipy import sparse

from sknetwork.clustering import Louvain, KMeans
from sknetwork.gnn import GNNClassifier
from sknetwork.utils import KMeansDense

from data import load_data, load_patterns, preprocess_data
from distances import pairwise_wd_distance
from summarization import get_pattern_summaries, get_summarized_biadjacency, get_summarized_graph
from utils import build_pattern_attributes, get_gensim_model, get_root_directory, pattern_attributes


# =================================================================
# Parameters
# =================================================================

#datasets = ['ingredients']
#datasets = ['london']
datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']
#datasets = ['sanFranciscoCrimes']
#datasets = ['lastfm']
sorted_attributes = True
with_prob = True
ss = [8, 7, 6, 5]
betas = [4]
delta = 10
#ss = [1]
#betas = [] 
avgs = defaultdict(dict)
INPATH = os.path.join(get_root_directory(), 'output/result')
if with_prob:
    INPATH += '/with_prob/attr_compressor_ratio'
MODEL_PATH = os.path.join(get_root_directory(), 'models/new_models')
#OUTPATH = os.path.join(INPATH, 'simpl_algo')
OUTPATH = INPATH

resolutions = {'wikivitals-fr': {1: 0, 3: 0, 5: 0.3, 6: 0.4, 8: 0.6, 9: 0.7, 10: 0.8, 11: 0.9, 12: 0.9, 14: 1.05, 15: 1.1,
                                 20: 1.4, 24: 1.8, 26: 1.86, 27: 1.89, 46: 2.95, 47: 3, 50: 3.4, 52: 3.5, 53: 3.8, 58: 3.87,
                                 64: 4, 66: 4.1, 85: 5.58},
                'wikivitals': {4: 0.5, 7: 0.8, 11: 1, 13: 1.3, 14: 1.18, 15: 1.4, 16: 1.5, 18: 1.6, 19: 1.6, 21: 1.76,
                               23: 1.83, 28: 2.3, 29: 2.4, 30: 2.45, 31: 2.5, 34: 2.7, 36: 2.8, 38: 3.05, 39: 3.1, 40: 3.2, 46: 3.7,
                               47: 3.903, 49: 3.95, 57: 4.2, 61: 4.5, 63: 4.6, 64: 4.6, 65: 4.70, 85: 5.58, 89: 5.7,
                               101: 6.4, 109: 6.7, 118: 7.25, 119: 7.25, 121: 7.3},
                'wikischools': {3: 0.4, 4: 0.45, 8: 1, 3: 0.4, 8: 0.58, 9: 0.9, 10: 1.18, 12: 1.3, 13: 1.4, 15: 1.5,
                                16: 1.58, 18: 1.69, 21: 1.9, 22: 1.95, 24: 2.1, 25: 2.2, 29: 2.3, 30: 2.3, 32: 2.51, 37: 2.6, 39: 2.75,
                                40: 2.75, 42: 2.8, 46: 2.99, 48: 3.05, 49: 3.1, 52: 3.3, 54: 3.4, 57: 3.7, 58: 3.7, 70: 4.2,
                                71: 4.4, 73: 4.41, 74: 4.45, 76: 4.5, 79: 4.59},
                'lastfm': {60: 2.74, 32: 0.65},
                'sanFranciscoCrimes': {5: 0.13, 6: 0.14},
                'london': {1: 0.05}}

# Run experiment
# ------------------------------------------------------------------

for dataset in datasets:
    avgs[dataset] = defaultdict(dict)

    # Load netset data
    adjacency, biadjacency, names, names_col, labels = load_data(dataset)

    # Gensim model
    model = get_gensim_model(MODEL_PATH, f'gensim_model_{dataset}', biadjacency, names_col)
    print(f'Model information: {model.wv}')

    for b in betas:
        avgs[dataset][b] = defaultdict(dict)
        for s in ss:
            avgs[dataset][b][s] = defaultdict(list)

            print(f'* Dataset: {dataset} - beta={b} - s={s}')

            # Preprocess data (get same attribute order as in UnexPattern)
            new_biadjacency, words = preprocess_data(biadjacency, names_col, s, sort_data=sorted_attributes)
            
            # Load Excess patterns
            patterns = load_patterns(dataset, b, s, sorted_attributes, INPATH, with_prob, delta)

            # Graph summarization
            # ------------------------------------------------------------------
            
            # Summarized adjacency and biadjacency
            summarized_adjacency = get_summarized_graph(adjacency, patterns)            
            summarized_biadjacency = get_summarized_biadjacency(adjacency, new_biadjacency, patterns)

            # Pattern summaries
            #pattern_summaries = get_pattern_summaries_new(patterns)
            #nb_p_s = len(pattern_summaries)
            labels_cc_summarized, mask = get_pattern_summaries(summarized_adjacency)
            nb_p_s = len(np.unique(labels_cc_summarized))

            # Baseline methods
            # ------------------------------------------------------------------

            # Louvain
            if nb_p_s > 1 and dataset != 'ingredients':
                louvain = Louvain(resolution=resolutions.get(dataset).get(nb_p_s)) 
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

            if len(labels) > 0:
                gnn.fit(adjacency, features, labels, train_size=0.8, val_size=0.1, test_size=0.1, n_epochs=50)
                # KMeans on GNN node embedding
                gnn_embedding = gnn.layers[-1].embedding
                kmeans = KMeansDense(n_clusters=nb_p_s) # k = number of connected components in summarized graph
                kmeans_gnn_labels = kmeans.fit_transform(gnn_embedding)

            # Spectral + KMeans
            kmeans = KMeans(n_clusters=nb_p_s) # k = number of connected components in summarized graph
            kmeans_spectral_labels = kmeans.fit_transform(adjacency)

            # KMeans model on d2v embeddings
            kmeans = KMeansDense(n_clusters=nb_p_s) # k = number of connected components in summarized graph
            kmeans_doc2vec_labels = kmeans.fit_transform(model.dv.vectors)
            
            # Concept x attributes matrices for each method
            # ------------------------------------------------------------------
            # TODO: Refactorization according to dataset choices
            
            # Only run wasserstein distances between patterns on small datasets
            #concept_attributes = np.zeros((len(patterns), biadjacency.shape[1]))
            #for i, c in enumerate(patterns[1:]):
            #    concept_attributes[i, c[1]] = 1

            concept_summarized_attributes = pattern_attributes(summarized_biadjacency, labels_cc_summarized, mask)

            if len(labels) > 0:
                if nb_p_s > 1:
                    concept_louvain_attributes, concept_gnn_kmeans_attributes, concept_spectral_kmeans_attributes, concept_doc2vec_kmeans_attributes = build_pattern_attributes(biadjacency, labels_louvain, kmeans_gnn_labels, kmeans_spectral_labels, kmeans_doc2vec_labels)
                    #concept_summarized_attributes = get_s_pattern_attributes(pattern_summaries, new_biadjacency.shape[1])
                else:
                    # Louvain cannot be applied
                    _, concept_gnn_kmeans_attributes, concept_spectral_kmeans_attributes, concept_doc2vec_kmeans_attributes = build_pattern_attributes(biadjacency, kmeans_gnn_labels, kmeans_gnn_labels, kmeans_spectral_labels, kmeans_doc2vec_labels)
                    #concept_summarized_attributes = get_s_pattern_attributes(pattern_summaries, new_biadjacency.shape[1])

            elif nb_p_s > 1 and dataset != 'ingredients':
                # GNN cannot be applied
                concept_louvain_attributes, _, concept_spectral_kmeans_attributes, concept_doc2vec_kmeans_attributes = build_pattern_attributes(biadjacency, labels_louvain, labels_louvain, kmeans_spectral_labels, kmeans_doc2vec_labels)        
                #concept_summarized_attributes = get_s_pattern_attributes(pattern_summaries, new_biadjacency.shape[1])
            else:
                # GNN and Louvain cannot be applied
                _, _, concept_spectral_kmeans_attributes, concept_doc2vec_kmeans_attributes = build_pattern_attributes(biadjacency, kmeans_spectral_labels, kmeans_spectral_labels, kmeans_spectral_labels, kmeans_doc2vec_labels)
                #concept_summarized_attributes = get_s_pattern_attributes(pattern_summaries, new_biadjacency.shape[1])           
            
            # Pairwise Wassertein distances between embeddings
            # ------------------------------------------------------------------
            #d2v_wd_matrix_concepts = pairwise_wd_distance(concept_attributes, len(patterns), model, words)
            d2v_wd_matrix_summarized = pairwise_wd_distance(concept_summarized_attributes, nb_p_s, model, words)
            if nb_p_s > 1 and dataset != 'ingredients':
                d2v_wd_matrix_louvain = pairwise_wd_distance(concept_louvain_attributes, nb_louvain, model, names_col)
            if len(labels) > 0:
                d2v_wd_matrix_gnn_kmeans = pairwise_wd_distance(concept_gnn_kmeans_attributes, nb_p_s, model, names_col)
            d2v_wd_matrix_spectral_kmeans = pairwise_wd_distance(concept_spectral_kmeans_attributes, nb_p_s, model, names_col)
            d2v_wd_matrix_d2v_kmeans = pairwise_wd_distance(concept_doc2vec_kmeans_attributes, nb_p_s, model, names_col)

            # Save result
            #with open(f'{OUTPATH}/wasserstein_distances_{dataset}_{b}_{s}_patterns.pkl', 'wb') as f:
            #    np.save(f, d2v_wd_matrix_concepts)
            with open(f'{OUTPATH}/wasserstein_distances_{dataset}_{b}_{s}_summaries.pkl', 'wb') as f:
                np.save(f, d2v_wd_matrix_summarized)
            if nb_p_s > 1 and dataset != 'ingredients':
                with open(f'{OUTPATH}/wasserstein_distances_{dataset}_{b}_{s}_louvain.pkl', 'wb') as f:
                    np.save(f, d2v_wd_matrix_louvain)
            if len(labels) > 0:
                with open(f'{OUTPATH}/wasserstein_distances_{dataset}_{b}_{s}_gnn_kmeans.pkl', 'wb') as f:
                    np.save(f, d2v_wd_matrix_gnn_kmeans)
            with open(f'{OUTPATH}/wasserstein_distances_{dataset}_{b}_{s}_spectral_kmeans.pkl', 'wb') as f:
                np.save(f, d2v_wd_matrix_spectral_kmeans)
            with open(f'{OUTPATH}/wasserstein_distances_{dataset}_{b}_{s}_d2v_kmeans.pkl', 'wb') as f:
                np.save(f, d2v_wd_matrix_d2v_kmeans)
