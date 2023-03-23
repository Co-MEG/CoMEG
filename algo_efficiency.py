from collections import defaultdict

from algorithm import run_unex_patterns
from compressors import generation_complexity, generation_complexity_attrs
from data import load_data


# ******************************************************** #
# Mine unexpected patterns
# ******************************************************** #

# =================================================================
# Parameters
# =================================================================

#datasets = ['wikihumans']
datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']
#datasets = ['wikischools']
#datasets = ['ingredients']
#datasets = ['sanFranciscoCrimes']
#datasets = ['lastfm']

betas = [4]
ss = [8, 7, 6, 5]
deltas = [0]
OUTPATH = '/Users/simondelarue/Documents/PhD/Research/Co-Meg/CoMEG/output/result/with_prob/attr_compressor'

order_attributes = [True]

nb_pattern_dict = defaultdict(dict) 
# -------------------------------------------------------

for dataset in datasets:
    print(f'**** {dataset}')

    nb_pattern_dict[dataset] = defaultdict(dict)

    # Load data
    adjacency, biadjacency, names, words, labels = load_data(dataset)

    for order_attr in order_attributes:
        print(f'---- {order_attr}')

        nb_pattern_dict[dataset][order_attr] = defaultdict(list)

        for beta in betas:
            print(f'== {beta}')

            for s in ss:
                print(f'- {s}')

                for delt in deltas:
                            
                    outfile = f'{dataset}_{str(beta)}_{str(s)}_order{str(order_attr)}_delta_{delt}'

                    # Compute generation complexities
                    complexity_gen_graphs = generation_complexity(adjacency, biadjacency, n_attrs=15, n_iter=300)
                    #complexity_gen_attrs = generation_complexity_attrs(adjacency, biadjacency, n_attrs=15, n_iter=300)

                    # Run algorithm
                    #nb_patterns = run_unex_patterns(adjacency, biadjacency, words, complexity_gen_graphs,
                                                    #complexity_gen_attrs, order_attr, s, beta, delt, outfile, OUTPATH)
                    nb_patterns = run_unex_patterns(adjacency, biadjacency, words, complexity_gen_graphs,
                                                    order_attr, s, beta, delt, outfile, OUTPATH)

                    print(f'Nombre de patterns: {nb_patterns}')
                    # Save number of patterns
                    nb_pattern_dict[dataset][order_attr][beta].append(nb_patterns)
                    #print(nb_pattern_dict)

#with open(f'number_of_patterns_history.pkl', 'wb') as f:
#   pickle.dump(nb_pattern_dict, f)
    