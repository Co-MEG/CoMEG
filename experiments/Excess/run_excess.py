from collections import defaultdict
import json
import os
import numpy as np
import sys

sys.path.append('../..')
from data import get_pw_distance_matrix, load_data, preprocess_data, read_parameters, get_excess_pattern, get_sias_pattern
from distances import pairwise_wd_distance
from metrics import coverage_excess, diversity, width_excess
from utils import get_gensim_model, get_root_directory


if __name__=='__main__':
    
    # Get parameters
    parameter_filename = sys.argv[1]
    parameters = read_parameters(parameter_filename)
    INPATH = os.path.join(os.getcwd(), parameters.get('patterns_path'))
    MODEL_PATH = os.path.join(get_root_directory(), 'models/new_models')
    
    informations_excess = defaultdict(list)
    
    for dataset in parameters.get('datasets'):
        print(f'****Dataset: {dataset}')

        # Load netset data
        adjacency, biadjacency, names, names_col, labels = load_data(dataset)

        # Gensim model
        model = get_gensim_model(MODEL_PATH, f'gensim_model_{dataset}', biadjacency, names_col)

        for s_param in parameters.get('s'):
            print(f'--s={s_param}')

            # Preprocess data (get same attribute order as in UnexPattern)
            new_biadjacency, words = preprocess_data(biadjacency, names_col, s_param, sort_data=False)

            # Load Excess patterns
            with open(f'{INPATH}/{dataset}/retrievedPatterns_{s_param}.json', 'rb') as f:
                data = json.load(f)

            nb_excess_patterns = len(data.get('patterns'))

            # Preprocess Excess patterns
            if dataset != 'sanFranciscoCrimes':
                excess_patterns = [get_sias_pattern(data.get('patterns')[idx]) for idx in range(nb_excess_patterns)]
                excess_patterns_filt = [p for p in excess_patterns if len(p[0]) >= s_param]
            else:
                excess_patterns = [get_excess_pattern(data.get('patterns')[idx], names, names_col) for idx in range(nb_excess_patterns)]
                excess_patterns_filt = [p for p in excess_patterns if len(p[0]) >= s_param]
            
            # Pattern x attributes matrix
            nb_excess_patterns_filt = len(excess_patterns_filt)
            if dataset != 'sanFranciscoCrimes':
                excess_patterns_attributes = np.zeros((nb_excess_patterns_filt, new_biadjacency.shape[1]))
            else:
                excess_patterns_attributes = np.zeros((nb_excess_patterns_filt, biadjacency.shape[1]))
            for i, p in enumerate(excess_patterns_filt):
                excess_patterns_attributes[i, p[1]] = 1

            # Wasserstein distances
            wd_filename = f'wasserstein_distances_{dataset}_5_{s_param}_excess_patterns.pkl'
            if os.path.exists(f'{INPATH}/{dataset}'):
                wd_distances_excess = get_pw_distance_matrix(dataset, beta=5, s=s_param, path=os.path.join(INPATH, dataset), method='excess_patterns')
            else:
                wd_distances_excess = pairwise_wd_distance(excess_patterns_attributes, nb_excess_patterns_filt, model, names_col)
                # Save distances
                with open(f'{INPATH}/{dataset}/{wd_filename}', 'wb') as f:
                    np.save(f, wd_distances_excess)

            # Information metric
            div = diversity(wd_distances_excess, gamma=parameters.get('gamma'))
            cov = coverage_excess(excess_patterns_filt, adjacency.shape[0])
            conc = width_excess(adjacency, biadjacency, nb_excess_patterns, excess_patterns)
            information = (div * cov) / conc
            informations_excess[dataset].append(information)
            
            with open(f"{INPATH}/{dataset}/information_details_{dataset}_5_{s_param}_{parameters.get('gamma')}excess.txt", 'w') as f:
                f.write(f'{div}, {cov}, {conc}, {information}')
            print(f'{div}, {cov}, {conc}, {information}')
