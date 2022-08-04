from src.algorithms import AdamicAdar
from src.graph import BipartiteGraph

from sknetwork.utils import get_neighbors as gn
from sknetwork.linkpred import JaccardIndex #, AdamicAdar
import numpy as np
import json
import os

# Data path
IN_PATH = 'goodreads_poetry'
USE_CACHE = True
TEST_SIZE = 0.3

if __name__ == '__main__':
    
    # Build graph
    g = BipartiteGraph()
    g.load_data(IN_PATH, use_cache=USE_CACHE)

    # Verif
    print(type(g))
    print(f"Number of users: {len(g.V.get('left'))}")
    print(f"Number of books: {len(g.V.get('right'))}")
    print(f"Number of reviews: {len(g.E)}")

    # Link prediction
    # ----------------------
    m = g.number_of_edges()

    TRAIN_TEST_SPLIT = False
    if TRAIN_TEST_SPLIT:
        print('Splitting graph into train-test...')
        train_g, test_g = g.train_test_split(test_size=TEST_SIZE)
    
    """print(f"Number of users in train-test: {len(train_g.V.get('left'))}-{len(test_g.V.get('left'))}")
    print(f"Number of books in train-test: {len(train_g.V.get('right'))}-{len(test_g.V.get('right'))}")
    print(f"Number of reviews in train-test: {len(train_g.E)}-{len(test_g.E)}")
    print(f"Number of edge attributes in train-test: {len(train_g.edge_attr)}-{len(test_g.edge_attr)}")
    """
    # Adamic Adar
    first_node = g.V['left'][0]
    if TRAIN_TEST_SPLIT:
        neighbors = train_g.get_neighbors(first_node)
    else:
        neighbors = g.get_neighbors(first_node)
    #neighbors_2hops = g.get_neighbors_2hops(first_node)
    print(f'# Neighbors of node: {first_node} : {len(neighbors)}')
    #print(f'# Neighbors of node from same set: {first_node} : {len(neighbors_2hops)}')
    
    print('Adamic Adar index...')
    aa = AdamicAdar()
    n_nodes_predict = 5
    if TRAIN_TEST_SPLIT:
        scores = aa.predict_sample(train_g, train_g.V['left'][:n_nodes_predict])
    else:
        scores = aa.predict_sample(g, g.V['left'][:n_nodes_predict])
    
    # Save results for index
    path = os.path.join(os.getcwd(), 'data', 'goodreads_poetry', 'result', 'adamic_adar.json')
    with open(path, "w") as f:
        json.dump(scores , f) 


    # Test first order metrics
        # select randomly x% of pairs of nodes in the original graph -> computationaly possible
        # select randomly y% of edges in the test set.
        # select randomly z% of edges in the train set.
        # compute index on these pairs
        # test with ground truth, i.e. compare score with the existence/absence of edge in the orignial graph
        # Compute ROC AUC curve


    #idx = np.where(train_g.names_row == first_node)[0][0]
    #scores = aa.predict(idx)
    #scores = aa.predict(np.arange(0, 377799))
    #print(scores.shape)
    #print(len(scores))
    #print(sorted(scores, reverse=True)[:10])
    #scores = aa.predict(train_g, transpose=False)
    #scores = aa.predict(g, transpose=False)
    #print(scores[:10])
    

    # SKNETWORK - VERIF
    """total = 0
    idx = np.where(g.names_row == first_node)[0][0]
    print(g.names_col[gn(g.adjacency_csr, idx)]) # OK
    for n in gn(g.adjacency_csr, idx):
        nn = len(g.names_row[gn(g.adjacency_csr, n, transpose=True)])
        print(nn) # OK
        total += nn
    print(total)"""
    