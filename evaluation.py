from collections import defaultdict
import os
import numpy as np
import pickle

from baselines import get_louvain, get_community_graph, get_gnn, get_spectral, get_doc2vec
from data import load_data, preprocess_data, load_patterns, get_pw_distance_matrix
from metrics import information_summaries
from summarization import get_pattern_summaries_new, get_summarized_graph, get_summarized_biadjacency, get_pattern_summaries
from utils import get_root_directory, pattern_attributes


# =================================================================
# Parameters
# =================================================================

#datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']
#datasets = ['london']
datasets = ['ingredients']
#datasets = ['lastfm']
#datasets = ['sanFranciscoCrimes']
betas = [1] # Beta: 4 for all datasets, except beta=1 for ingredients
gamma = 0.05 # Wikipedia datasets: 0.8, sanFranciscoCrimes: 0.2, ingredients: 0.05
ss = [8, 7, 6, 5]
#INPATH = os.path.join(get_root_directory(), 'output/result/with_prob')
#OUTPATH = os.path.join(INPATH, 'new')
INPATH = os.path.join(get_root_directory(), 'output/result/with_prob/simpl_algo')
OUTPATH = INPATH

with_order = [True]
with_prob = True
informations = defaultdict()
new_summaries = False

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
            patterns = load_patterns(dataset, b, s, with_order[0], INPATH, with_prob)
            
            # Summarized graph + features
            summarized_adj = get_summarized_graph(adjacency, patterns)
            summarized_biadj = get_summarized_biadjacency(adjacency, new_biadjacency, patterns)
            
            # Pattern summaries 
            pattern_summary_labels, mask = get_pattern_summaries(summarized_adj)
            n_p_summaries = len(np.unique(pattern_summary_labels))
            print(f'# pattern summaries: {n_p_summaries}')
            pattern_summarized_attributes = pattern_attributes(summarized_biadj, pattern_summary_labels, mask)

            # NEW (without connected components)
            if new_summaries:
                pattern_summaries = get_pattern_summaries_new(patterns)
                n_p_summaries = len(pattern_summaries)
                pattern_summarized_attributes = np.zeros((n_p_summaries, biadjacency.shape[1]))
                for i, p_s in enumerate(pattern_summaries):
                    for attr in p_s[1]:
                        pattern_summarized_attributes[i, attr] += 1
                print(f'# pattern summaries new: {n_p_summaries}')
            
            # Louvain
            if n_p_summaries > 1 and dataset != 'ingredients':
                louvain_labels = get_louvain(dataset, adjacency, n_p_summaries)
                pattern_louvain_attributes = pattern_attributes(biadjacency, louvain_labels)
                louvain_adj = get_community_graph(adjacency, louvain_labels)
                n_p_louvain = len(np.unique(louvain_labels))

            # GNN
            if len(labels) > 0:
                gnn_labels = get_gnn(adjacency, biadjacency, labels, 16, n_p_summaries)
                pattern_gnn_attributes = pattern_attributes(biadjacency, gnn_labels)
                gnn_adj = get_community_graph(adjacency, gnn_labels)

            # Spectral
            spectral_labels = get_spectral(adjacency, n_p_summaries)
            pattern_spectral_attributes = pattern_attributes(biadjacency, spectral_labels)
            spectral_adj = get_community_graph(adjacency, spectral_labels)

            # Doc2Vec
            d2v_labels = get_doc2vec(biadjacency, dataset, names_col, n_p_summaries)
            pattern_d2v_attributes = pattern_attributes(biadjacency, d2v_labels)
            d2v_adj = get_community_graph(adjacency, d2v_labels)
            
            # Pariwise distances 
            #pw_distances_patterns = get_pw_distance_matrix(dataset, b, s, OUTPATH, method='patterns')
            pw_distances_summaries = get_pw_distance_matrix(dataset, b, s, OUTPATH, method='summaries')
            if n_p_summaries > 1 and dataset != 'ingredients':
                pw_distances_louvain = get_pw_distance_matrix(dataset, b, s, OUTPATH, method='louvain')
            if len(labels) > 0:
                pw_distances_gnn = get_pw_distance_matrix(dataset, b, s, OUTPATH, method='gnn_kmeans')
            pw_distances_spectral = get_pw_distance_matrix(dataset, b, s, OUTPATH, method='spectral_kmeans')
            pw_distances_d2v = get_pw_distance_matrix(dataset, b, s, OUTPATH, method='d2v_kmeans')
            
            # SG information
            #if new_summaries:
            #    print(f'    Patterns')
            #    information_patterns = information_summaries(adjacency, biadjacency, summarized_adj, pattern_summary_labels, n_p_summaries, #pattern_summarized_attributes, pw_distances_patterns, dataset, b, s, gamma, 'patterns', OUTPATH, patterns)

            print(f'    Summaries')
            if new_summaries:
                information_p_summaries = information_summaries(adjacency, biadjacency, summarized_adj, pattern_summary_labels, n_p_summaries, pattern_summarized_attributes, pw_distances_summaries, dataset, b, s, gamma, 'summaries', OUTPATH, pattern_summaries)
            else:
                information_p_summaries = information_summaries(adjacency, biadjacency, summarized_adj, pattern_summary_labels, n_p_summaries, pattern_summarized_attributes, pw_distances_summaries, dataset, b, s, gamma, 'summaries', OUTPATH)
            #information_summaries = information(summarized_adj, summarized_biadj, pw_distances_summaries)
            
            if n_p_summaries > 1 and dataset != 'ingredients':
                print(f'    Louvain')
                information_louvain = information_summaries(adjacency, biadjacency, louvain_adj, louvain_labels, n_p_louvain, pattern_louvain_attributes, pw_distances_louvain, dataset, b, s, gamma, 'louvain', OUTPATH)
                #information_louvain = information(louvain_adj, pattern_louvain_attributes, pw_distances_louvain)
            
            if len(labels) > 0:
                print(f'    GNN')
                information_gnn = information_summaries(adjacency, biadjacency, gnn_adj, gnn_labels, n_p_summaries, pattern_gnn_attributes, pw_distances_gnn, dataset, b, s, gamma, 'gnn', OUTPATH)
            #information_gnn = information(gnn_adj, pattern_gnn_attributes, pw_distances_gnn)
            print(f'    Spectral')
            information_spectral = information_summaries(adjacency, biadjacency, spectral_adj, spectral_labels, n_p_summaries, pattern_spectral_attributes, pw_distances_spectral, dataset, b, s, gamma, 'spectral', OUTPATH)
            #information_spectral = information(spectral_adj, pattern_spectral_attributes, pw_distances_spectral)
            print(f'    Doc2Vec')
            information_d2v = information_summaries(adjacency, biadjacency, d2v_adj, d2v_labels, n_p_summaries, pattern_d2v_attributes, pw_distances_d2v, dataset, b, s, gamma, 'd2v', OUTPATH)
            #information_d2v = information(d2v_adj, pattern_d2v_attributes, pw_distances_d2v)
            
            # Save information in dict
            informations[dataset][b]['summaries'].append(information_p_summaries)
            if n_p_summaries > 1 and dataset != 'ingredients':
                informations[dataset][b]['louvain'].append(information_louvain)
            if len(labels) > 0:
                informations[dataset][b]['gnn'].append(information_gnn)
            informations[dataset][b]['spectral'].append(information_spectral)
            informations[dataset][b]['doc2vec'].append(information_d2v)

# Save result
with open(f'{OUTPATH}/informations_evaluation.pkl', 'wb') as f:
    pickle.dump(informations, f)
    