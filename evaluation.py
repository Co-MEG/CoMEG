from collections import defaultdict
import numpy as np
import pickle

from baselines import get_louvain, get_community_graph, get_gnn, get_spectral, get_doc2vec
from data import load_data, preprocess_data, load_patterns, get_pw_distance_matrix
from metrics import information
from summarization import get_summarized_graph, get_summarized_biadjacency, get_pattern_summaries
from utils import pattern_attributes


# =================================================================
# Parameters
# =================================================================

datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']
#datasets = ['lastfm']
betas = [8, 7, 6, 5]
ss = [8, 7, 6, 5]

with_order = [True]
delta = 0.2
informations = defaultdict()

# =================================================================
# Evaluation
# =================================================================

for d, dataset in enumerate(datasets):
    informations[dataset] = defaultdict(dict)
    # Load data
    adjacency, biadjacency, names, names_col, labels = load_data(dataset)
    print(f'**Dataset: {dataset}...')
        
    for i, b in enumerate(betas):
        print(f' ==Beta: {b}')
        informations[dataset][b] = defaultdict(list)
        for k, s in enumerate(ss):
            print(f'  ---s: {s}')
            
            new_biadjacency, words = preprocess_data(biadjacency, names_col, s)
            
            # Load patterns
            patterns = load_patterns(dataset, b, s, True)
            
            # Summarized graph + features
            summarized_adj = get_summarized_graph(adjacency, patterns)
            summarized_biadj = get_summarized_biadjacency(adjacency, new_biadjacency, patterns)
            
            # Pattern summaries 
            pattern_summary_labels, mask = get_pattern_summaries(summarized_adj)
            n_p_summaries = len(np.unique(pattern_summary_labels))
            
            # Louvain
            louvain_labels = get_louvain(dataset, adjacency, n_p_summaries)
            pattern_louvain_attributes = pattern_attributes(biadjacency, louvain_labels)
            louvain_adj = get_community_graph(adjacency, louvain_labels)

            # GNN
            gnn_labels = get_gnn(adjacency, biadjacency, labels, 16, n_p_summaries)
            pattern_gnn_attributes = pattern_attributes(biadjacency, gnn_labels)
            gnn_adj = get_community_graph(adjacency, gnn_labels)

            # Spectral
            spectral_labels = get_spectral(adjacency, n_p_summaries)
            pattern_spectral_attributes = pattern_attributes(biadjacency, spectral_labels)
            spectral_adj = get_community_graph(adjacency, spectral_labels)

            # Doc2Vec
            d2v_labels = get_doc2vec(biadjacency, names_col, n_p_summaries)
            pattern_d2v_attributes = pattern_attributes(biadjacency, d2v_labels)
            d2v_adj = get_community_graph(adjacency, d2v_labels)
            
            # Pariwise distances 
            pw_distances_summaries = get_pw_distance_matrix(dataset, b, s, method='summaries')
            pw_distances_louvain = get_pw_distance_matrix(dataset, b, s, method='louvain')
            pw_distances_gnn = get_pw_distance_matrix(dataset, b, s, method='gnn_kmeans')
            pw_distances_spectral = get_pw_distance_matrix(dataset, b, s, method='spectral_kmeans')
            pw_distances_d2v = get_pw_distance_matrix(dataset, b, s, method='d2v_kmeans')
            
            # SG information
            print(f'    Summaries')
            information_summaries = information(summarized_adj, summarized_biadj, pw_distances_summaries)
            print(f'    Louvain')
            information_louvain = information(louvain_adj, pattern_louvain_attributes, pw_distances_louvain)
            print(f'    GNN')
            information_gnn = information(gnn_adj, pattern_gnn_attributes, pw_distances_gnn)
            print(f'    Spectral')
            information_spectral = information(spectral_adj, pattern_spectral_attributes, pw_distances_spectral)
            print(f'    Doc2Vec')
            information_d2v = information(d2v_adj, pattern_d2v_attributes, pw_distances_d2v)
            
            # Save information in dict
            informations[dataset][b]['summaries'].append(information_summaries)
            informations[dataset][b]['louvain'].append(information_louvain)
            informations[dataset][b]['gnn'].append(information_gnn)
            informations[dataset][b]['spectral'].append(information_spectral)
            informations[dataset][b]['doc2vec'].append(information_d2v)

# Save result
with open(f'informations_evaluation_new.pkl', 'wb') as f:
    pickle.dump(informations, f)
    