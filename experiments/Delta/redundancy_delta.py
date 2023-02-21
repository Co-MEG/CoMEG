from collections import defaultdict
import os
import gensim
import numpy as np
import pickle
import sys

from sknetwork.clustering import Louvain, KMeans
from sknetwork.gnn import GNNClassifier
from sknetwork.utils import get_degrees, KMeansDense

sys.path.append('../..')

from corpus import MyCorpus
from data import load_data
from distances import pairwise_wd_distance
from summarization import get_pattern_summaries, get_summarized_biadjacency, get_summarized_graph
from utils import build_pattern_attributes, get_gensim_model, get_root_directory, load_gensim_model, pattern_attributes, save_gensim_model


def load_result(path, filename):
    with open(f'{path}/{filename}.bin', 'rb') as data:
        result = pickle.load(data)
    return result

datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']
sorted_attributes = True
with_prob = True
ss = [5]
betas = [5]
delta = 1 # To be done: all datasets [3, 2, 1]
outpath = f'delta_{delta}'
MODEL_PATH = os.path.join(get_root_directory(), 'models/new_models')

avgs = defaultdict(dict)

# Run experiment
# ------------------------------------------------------------------
for dataset in datasets:
    avgs[dataset] = defaultdict(dict)

    # Load and preprocess data
    adjacency, biadjacency, names, words, labels = load_data(dataset)
    orig_words = words.copy()

    # Gensim model
    model = get_gensim_model(MODEL_PATH, f'gensim_model_{dataset}', biadjacency, orig_words)

    for b in betas:
        avgs[dataset][b] = defaultdict(dict)
        for s in ss:
            avgs[dataset][b][s] = defaultdict(list)

            print(f'* Dataset: {dataset} - beta={b} - s={s}')
            
            # Load result
            # ------------------------------------------------------------------
            if with_prob:
                filename = f'result_{dataset}_{b}_{s}_order{str(sorted_attributes)}_delta_{delta}'
            else:
                filename = f'result_{dataset}_{b}_{s}_order{str(sorted_attributes)}'
            result = load_result(outpath, filename)
            
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
            # Summarized adjacency and biadjacency
            summarized_adjacency = get_summarized_graph(adjacency, result)            
            summarized_biadjacency = get_summarized_biadjacency(adjacency, filt_biadjacency, result)
            # Pattern summaries
            labels_cc_summarized, mask = get_pattern_summaries(summarized_adjacency)
            nb_cc = len(np.unique(labels_cc_summarized))

            # patterns from Unexpectedness algorithm 
            concept_attributes = np.zeros((len(result), biadjacency.shape[1]))
            for i, c in enumerate(result[1:]):
                concept_attributes[i, c[1]] = 1
            
            # Pattern x attributes matrices for all methods
            concept_summarized_attributes = pattern_attributes(biadjacency, labels_cc_summarized)                                                        
            
            # Pairwise Wassertein distances between embeddings
            # ------------------------------------------------------------------
            d2v_wd_matrix_concepts = pairwise_wd_distance(concept_attributes, nb_cc, model, sorted_names_col)
            d2v_wd_matrix_summarized = pairwise_wd_distance(concept_summarized_attributes, nb_cc, model, sorted_names_col)
            
            # Save result
            with open(f'{outpath}/wasserstein_distances_{dataset}_{b}_{s}_patterns.pkl', 'wb') as f:
                np.save(f, d2v_wd_matrix_concepts)
            with open(f'{outpath}/wasserstein_distances_{dataset}_{b}_{s}_summaries.pkl', 'wb') as f:
                np.save(f, d2v_wd_matrix_summarized)
            