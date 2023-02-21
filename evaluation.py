from collections import defaultdict
import numpy as np
import pickle

from baselines import get_louvain, get_community_graph, get_gnn, get_spectral, get_doc2vec
from data import load_data, preprocess_data, load_patterns, get_pw_distance_matrix
from metrics import information, information_summaries
from summarization import get_summarized_graph, get_summarized_biadjacency, get_pattern_summaries
from utils import pattern_attributes


# =================================================================
# Parameters
# =================================================================

#datasets = ['sanFranciscoCrimes']
#datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']
#datasets = ['london']
datasets = ['ingredients']
#datasets = ['lastfm']
betas = [1]
ss = [8, 7, 6, 5]
inpath = '/Users/simondelarue/Documents/PhD/Research/Co-Meg/CoMEG/output/result/with_prob'

with_order = [True]
with_prob = True
gamma = 0.5
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
            patterns = load_patterns(dataset, b, s, with_order[0], inpath, with_prob)
            
            # Summarized graph + features
            summarized_adj = get_summarized_graph(adjacency, patterns)
            summarized_biadj = get_summarized_biadjacency(adjacency, new_biadjacency, patterns)
            
            # Pattern summaries 
            pattern_summary_labels, mask = get_pattern_summaries(summarized_adj)
            n_p_summaries = len(np.unique(pattern_summary_labels))
            print(f'# pattern summaries: {n_p_summaries}')
            pattern_summarized_attributes = pattern_attributes(summarized_biadj, pattern_summary_labels, mask)
            
            # Louvain
            if dataset != 'ingredients':
                louvain_labels = get_louvain(dataset, adjacency, n_p_summaries)
                pattern_louvain_attributes = pattern_attributes(biadjacency, louvain_labels)
                louvain_adj = get_community_graph(adjacency, louvain_labels)
                n_p_louvain = len(np.unique(louvain_labels))

            # GNN
            if dataset != 'sanFranciscoCrimes' and dataset != 'london' and dataset != 'ingredients':
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
            pw_distances_summaries = get_pw_distance_matrix(dataset, b, s, inpath + '/new', method='summaries')
            if dataset != 'ingredients':
                pw_distances_louvain = get_pw_distance_matrix(dataset, b, s, inpath + '/new', method='louvain')
            if dataset != 'sanFranciscoCrimes' and dataset != 'london' and dataset != 'ingredients':
                pw_distances_gnn = get_pw_distance_matrix(dataset, b, s, inpath + '/new', method='gnn_kmeans')
            pw_distances_spectral = get_pw_distance_matrix(dataset, b, s, inpath + '/new', method='spectral_kmeans')
            pw_distances_d2v = get_pw_distance_matrix(dataset, b, s, inpath + '/new', method='d2v_kmeans')
            
            # SG information
            print(f'    Summaries')
            information_p_summaries = information_summaries(adjacency, biadjacency, summarized_adj, pattern_summary_labels, n_p_summaries, pattern_summarized_attributes, pw_distances_summaries, dataset, b, s, gamma, 'summaries', inpath + '/new')
            #information_summaries = information(summarized_adj, summarized_biadj, pw_distances_summaries)
            if dataset != 'ingredients':
                print(f'    Louvain')
                information_louvain = information_summaries(adjacency, biadjacency, louvain_adj, louvain_labels, n_p_louvain, pattern_louvain_attributes, pw_distances_louvain, dataset, b, s, gamma, 'louvain', inpath + '/new')
                #information_louvain = information(louvain_adj, pattern_louvain_attributes, pw_distances_louvain)
            
            if dataset != 'sanFranciscoCrimes' and dataset != 'london' and dataset != 'ingredients':
                print(f'    GNN')
                information_gnn = information_summaries(adjacency, biadjacency, gnn_adj, gnn_labels, n_p_summaries, pattern_gnn_attributes, pw_distances_gnn, dataset, b, s, gamma, 'gnn', inpath + '/new')
            #information_gnn = information(gnn_adj, pattern_gnn_attributes, pw_distances_gnn)
            print(f'    Spectral')
            information_spectral = information_summaries(adjacency, biadjacency, spectral_adj, spectral_labels, n_p_summaries, pattern_spectral_attributes, pw_distances_spectral, dataset, b, s, gamma, 'spectral', inpath + '/new')
            #information_spectral = information(spectral_adj, pattern_spectral_attributes, pw_distances_spectral)
            print(f'    Doc2Vec')
            information_d2v = information_summaries(adjacency, biadjacency, d2v_adj, d2v_labels, n_p_summaries, pattern_d2v_attributes, pw_distances_d2v, dataset, b, s, gamma, 'd2v', inpath + '/new')
            #information_d2v = information(d2v_adj, pattern_d2v_attributes, pw_distances_d2v)
            
            # Save information in dict
            informations[dataset][b]['summaries'].append(information_p_summaries)
            if dataset != 'ingredients':
                informations[dataset][b]['louvain'].append(information_louvain)
            if dataset != 'sanFranciscoCrimes' and dataset != 'london' and dataset != 'ingredients':
                informations[dataset][b]['gnn'].append(information_gnn)
            informations[dataset][b]['spectral'].append(information_spectral)
            informations[dataset][b]['doc2vec'].append(information_d2v)

# Save result
with open(f'{inpath}/new/informations_evaluation_new_conc.pkl', 'wb') as f:
    pickle.dump(informations, f)
    