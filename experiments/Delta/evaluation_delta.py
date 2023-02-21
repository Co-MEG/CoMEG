import numpy as np
import os
import sys

sys.path.append('../..')

from data import load_data, preprocess_data, load_patterns, get_pw_distance_matrix
from metrics import information_summaries
from summarization import get_summarized_graph, get_summarized_biadjacency, get_pattern_summaries
from utils import pattern_attributes


# =================================================================
# Parameters
# =================================================================

datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']
betas = [5]
ss = [5]
INPATH = '/Users/simondelarue/Documents/PhD/Research/Co-Meg/CoMEG/experiments/Delta/'
deltas = [4, 3, 2, 1, 0]

with_order = [True]
with_prob = True
gamma = 0.8

# =================================================================
# Evaluation
# =================================================================

for d, dataset in enumerate(datasets):
    
    # Load data
    adjacency, biadjacency, names, names_col, labels = load_data(dataset)
    print(f'**Dataset: {dataset}...')
        
    for i, b in enumerate(betas):
        print(f' ==Beta: {b}')

        for k, s in enumerate(ss):
            print(f'  ---s: {s}')

            for delt in deltas:
                print(f'  ---delta: {delt}')

                inpath = os.path.join(INPATH, f'delta_{delt}')
                new_biadjacency, words = preprocess_data(biadjacency, names_col, s)
            
                # Load patterns
                patterns = load_patterns(dataset, b, s, with_order[0], inpath, with_prob, delta=delt)
            
                # Summarized graph + features
                summarized_adj = get_summarized_graph(adjacency, patterns)
                summarized_biadj = get_summarized_biadjacency(adjacency, new_biadjacency, patterns)
                
                # Pattern summaries 
                pattern_summary_labels, mask = get_pattern_summaries(summarized_adj)
                n_p_summaries = len(np.unique(pattern_summary_labels))
                print(f'# pattern summaries: {n_p_summaries}')
                pattern_summarized_attributes = pattern_attributes(summarized_biadj, pattern_summary_labels, mask)
            
                # Pariwise distances 
                pw_distances_summaries = get_pw_distance_matrix(dataset, b, s, inpath, method='summaries', delta=delt)
                
                # SG information
                print(f'    Summaries')
                information_p_summaries = information_summaries(adjacency, biadjacency, summarized_adj, pattern_summary_labels, n_p_summaries, pattern_summarized_attributes, pw_distances_summaries, dataset, b, s, gamma, 'summaries', inpath)
