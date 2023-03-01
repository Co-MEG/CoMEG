from collections import defaultdict

from algorithm import run_unex_patterns
from compressors import generation_complexity
from data import load_data


# ******************************************************** #
# Mine unexpected patterns
# ******************************************************** #

# =================================================================
# Parameters
# =================================================================

#datasets = ['wikihumans']
#datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']
datasets = ['ingredients']
#datasets = ['sanFranciscoCrimes']
#datasets = ['lastfm']

betas = [4]
ss = [8, 7, 6, 5]
deltas = [0]
OUTPATH = '/Users/simondelarue/Documents/PhD/Research/Co-Meg/CoMEG/output/result/with_prob/old'

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
                for delt in deltas:
                            
                    outfile = f'{dataset}_{str(beta)}_{str(s)}_order{str(order_attr)}_delta_{delt}'

                    # Load data
                    adjacency, biadjacency, names, words, labels = load_data(dataset)

                    # Compute generation complexities
                    complexity_gen_graphs = generation_complexity(adjacency, biadjacency, n_attrs=15, n_iter=300)
                    
                    # Run algorithm
                    nb_patterns = run_unex_patterns(adjacency, biadjacency, words, complexity_gen_graphs, order_attr, s, beta, delt, outfile, OUTPATH)

                    # Save number of patterns
                    nb_pattern_dict[dataset][order_attr][beta].append(nb_patterns)
                    print(nb_pattern_dict)  

#with open(f'number_of_patterns_history.pkl', 'wb') as f:
#   pickle.dump(nb_pattern_dict, f)
    