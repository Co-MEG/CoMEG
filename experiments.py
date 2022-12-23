from collections import defaultdict
import numpy as np
import pickle
from scipy import sparse
from tqdm import tqdm

from sknetwork.data import load_netset
from sknetwork.utils import get_degrees

from algorithm import run_comeg
from compressors import mdl_graph
from derivation import extension_csc


# ******************************************************** #
# Run experiments
# ******************************************************** #

# -------------------------------------------------------
# Parameters
datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']

betas = [8, 7, 6, 5]
ss = [8, 7, 6, 5]

order_attributes = [False, True]

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

                graph = load_netset(dataset)
                adjacency = graph.adjacency
                biadjacency = graph.biadjacency
                names = graph.names
                words = graph.names_col
                labels = graph.labels

                print(f'Generation complexities for graph structure...')
                # Graph structure generation complexity
                attrs_degrees = get_degrees(biadjacency, transpose=True) / sum(get_degrees(biadjacency, transpose=True))
                attrs_indexes = np.arange(0, biadjacency.shape[1])
                complexity_gen_graphs = defaultdict(list)
                biadjacency_csc = biadjacency.tocsc()

                for i in tqdm(range(300)):
                    for num_a in range(15):
                        sel_attrs = np.random.choice(attrs_indexes, size=num_a, replace=False, p=attrs_degrees)
                        sel_nodes = extension_csc(sel_attrs, biadjacency_csc)
                        #sel_nodes = extension(sel_attrs, biadjacency)
                        sel_g = adjacency[sel_nodes, :][:, sel_nodes].astype(bool) + sparse.identity(len(sel_nodes)).astype(bool)
                        mdl = mdl_graph(sel_g)
                        #mdl = np.log2(len(sel_nodes)) # mdl is just the complexity of the number of nodes
                        if mdl != np.inf and len(sel_nodes) > 0:
                            complexity_gen_graphs[len(sel_nodes)].append(mdl)

                nb_patterns = run_comeg(adjacency, biadjacency, words, complexity_gen_graphs, order_attr, s, beta, outfile)

                # Save number of patterns
                #nb_pattern_dict[dataset][order_attr][beta].append(nb_patterns)
                nb_pattern_dict[dataset][order_attr][beta].append(nb_patterns)
                print(nb_pattern_dict)

with open(f'number_of_patterns_history.pkl', 'wb') as f:
    pickle.dump(nb_pattern_dict, f)
