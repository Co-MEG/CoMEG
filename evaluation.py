from collections import defaultdict
import numpy as np
import pickle

from baselines import get_louvain, get_community_graph, get_gnn
from data import load_data, preprocess_data, load_patterns, get_pw_distance_matrix
from metrics import information
from summarization import get_summarized_graph, get_summarized_biadjacency, get_pattern_summaries
from utils import pattern_attributes


# =================================================================
# Parameters
# =================================================================

datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']
methods = ['summaries', 'louvain', 'gnn']
betas = [8, 7, 6, 5]
ss = [8, 7, 6, 5]
with_order = [True]
delta = 0.2
informations = defaultdict()

for d, dataset in enumerate(datasets):
    informations[dataset] = defaultdict(dict)
    # Load data
    adjacency, biadjacency, names, names_col, labels = load_data(dataset)
    print(f'Dataset: {dataset}...')
        
    for i, b in enumerate(betas):
        informations[dataset][b] = defaultdict(list)
        for k, s in enumerate(ss):
            
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
            gnn_labels = get_gnn(dataset, adjacency, biadjacency, labels, 16, n_p_summaries)
            pattern_gnn_attributes = pattern_attributes(biadjacency, gnn_labels)
            gnn_adj = get_community_graph(adjacency, gnn_labels)
            
            # Pariwise distances 
            pw_distances_summaries = get_pw_distance_matrix(dataset, b, s, method='summaries')
            pw_distances_louvain = get_pw_distance_matrix(dataset, b, s, method='louvain')
            pw_distances_gnn = get_pw_distance_matrix(dataset, b, s, method='gnn_kmeans')
            
            # SG information
            information_summaries = information(summarized_adj, summarized_biadj, pw_distances_summaries)
            information_louvain = information(louvain_adj, pattern_louvain_attributes, pw_distances_louvain)
            information_gnn = information(gnn_adj, pattern_gnn_attributes, pw_distances_gnn)
            
            informations[dataset][b]['summaries'].append(information_summaries)
            informations[dataset][b]['louvain'].append(information_louvain)
            informations[dataset][b]['gnn'].append(information_gnn)

# Save result
with open(f'informations_evaluation.pkl', 'wb') as f:
    pickle.dump(informations, f)
    