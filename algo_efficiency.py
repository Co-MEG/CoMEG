from collections import defaultdict
import numpy as np
import pickle
from scipy import sparse
from tqdm import tqdm

from sknetwork.utils import get_degrees

from algorithm import run_unex_patterns
from compressors import mdl_graph, generation_complexity
from data import load_data
from derivation import extension_csc


# ******************************************************** #
# Run experiments
# ******************************************************** #

# -------------------------------------------------------
# Parameters
#datasets = ['wikihumans']
#datasets = ['wikivitals-fr', 'wikischools', 'wikivitals']
datasets = ['lastfm']

betas = [8, 7, 6, 5]
ss = [8, 7, 6, 5]

order_attributes = [True]

nb_pattern_dict = defaultdict(dict) 
# -------------------------------------------------------

for dataset in datasets:
    print(f'**** {dataset}')
    nb_pattern_dict[dataset] = defaultdict(dict)
    for order_attr in order_attributes:
        print(f'---- {order_attr}')
        nb_pattern_dict[dataset][order_attr] = defaultdict(list)
        for beta in betas:
            print(f'== {beta}')
            for s in ss:
                print(f'- {s}')
                            
                outfile = f'{dataset}_{str(beta)}_{str(s)}_order{str(order_attr)}'

                # Load data
                adjacency, biadjacency, names, words, labels = load_data(dataset)

                # Compute generation complexities
                complexity_gen_graphs = generation_complexity(adjacency, biadjacency, n_attrs=15, n_iter=300)
                
                # Run algorithm
                nb_patterns = run_unex_patterns(adjacency, biadjacency, words, complexity_gen_graphs, order_attr, s, beta, outfile, names)

                # Save number of patterns
                nb_pattern_dict[dataset][order_attr][beta].append(nb_patterns)
                print(nb_pattern_dict)  

with open(f'number_of_patterns_history.pkl', 'wb') as f:
    pickle.dump(nb_pattern_dict, f)
    