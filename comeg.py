from matplotlib.style import context
from src.algorithms import AdamicAdar
from src.concept_lattice import ConceptLattice
from src.context import FormalContext
from src.graph import BipartiteGraph

from sknetwork.utils import get_neighbors as gn
from sknetwork.linkpred import JaccardIndex #, AdamicAdar
import numpy as np
import json
import os

from sklearn import metrics

import matplotlib.pyplot as plt

# Data path
IN_PATH = 'goodreads_poetry'
USE_CACHE = True
TEST_SIZE = 0.05
TRAIN_TEST_SPLIT = True
PATH_RES = os.path.join(os.getcwd(), 'data', 'goodreads_poetry', 'result')


def plot_roc_auc(y_true, y_pred, ax):
    fpr, tpr, thresholds = metrics.roc_curve(y_true,  y_pred) 
    auc = metrics.roc_auc_score(y_true, y_pred)

    ax.plot(fpr, tpr, label='Adamic Adar')
    ax.set_title(f'ROC AUC Curve - AUC = {auc:.4f}', weight='bold')
    ax.legend()

def run_toy():
    # Toy context
    context_toy = FormalContext.from_csv(os.path.join(os.getcwd(), 'data/toy/digits.csv'))
    print(context_toy.I.todense().shape)
    print(context_toy.G)
    print(context_toy.M)
    print()

    print(context_toy.intention(['01', '03', '09']))
    # Concept Lattice
    L = ConceptLattice.from_context(context_toy)


def run():
    # Build graph
    g = BipartiteGraph()
    g.load_data(IN_PATH, use_cache=USE_CACHE)
    m = g.number_of_edges()

    # Verif
    print(type(g))
    print(f"Number of users: {len(g.V.get('left'))}")
    print(f"Number of books: {len(g.V.get('right'))}")
    print(f"Number of reviews: {len(g.E)}")
    
    
    # Link prediction
    # ----------------------
    #if TRAIN_TEST_SPLIT:
    #    print('Splitting graph into train-test...')
    #    train_g, test_g = g.train_test_split(test_size=TEST_SIZE)
    #    print(f'Number of reviews in test: {len(test_g.E)}')
    #    print(f'Number of left nodes used: {len(set([e[0] for e in test_g.E]))}')
    #    print(f'Number of right nodes used: {len(set([e[1] for e in test_g.E]))}')

    """# Adamic Adar
    print('Adamic Adar index...')
    aa = AdamicAdar()

    # predict edges
    scores = aa.predict_edges(train_g, test_g.E, transpose=False)
    print(f'Number of predictions: {len(scores)}')
    # Save results for AA index
    res = os.path.join(PATH_RES, 'adamic_adar_test_graph.json')
    with open(res, "w") as f:
        json.dump(scores , f) """

    # predict nodes
    """
    n_nodes_predict = 100
    first_node = g.V['left'][0]
    if TRAIN_TEST_SPLIT:
        neighbors = train_g.get_neighbors(first_node)
        scores = aa.predict_sample(train_g, train_g.V['left'][:n_nodes_predict])
    else:
        neighbors = g.get_neighbors(first_node)
        scores = aa.predict_sample(g, g.V['left'][:n_nodes_predict])
    
    # Save results for AA index
    with open(PATH_RES, "w") as f:
        json.dump(scores , f) 
    """
    
    # Evaluation
    # ----------

    #res = os.path.join(PATH_RES, 'adamic_adar.json')
    #results = json.load(open(res))

    # ground truth label
    #y_pred, y_true = zip(*[(values[2], g.has_edge(values[0], values[1])) for _, values in results.items()])
    #print(f'Number of predicted edges: {len(y_pred)}')

    # Plot results
    #fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    #plot_roc_auc(y_true, y_pred, ax)
    #plt.show()

    # Formal Concept Analysis
    # -----------------------

    # Subgraph in the vicinity of an edge
    res = os.path.join(PATH_RES, 'adamic_adar.json')
    results = json.load(open(res))

    pred_scores, pred_edges = zip(*[(values[2], (values[0], values[1])) for _, values in results.items()])
    max_score_edge = pred_edges[np.argmax(pred_scores)]
    print(f'Predicted edge: {max_score_edge}')
    
    subgraph = g.subgraph_vicinity(max_score_edge)
    print(f"# nodes in subgraph: {len(subgraph.V['left'])}, {len(subgraph.V['right'])}")
    print(f'# edges in subgraph: {len(subgraph.E)}')
    print(f"# of edge attributes: {len(subgraph.edge_attr)}")
    print(f"# of node attributes right: {len(subgraph.node_attr['right'])}")
    print(f'Edge exists in graph: {g.has_edge(u=max_score_edge[0], v=max_score_edge[1])}')

    #print(subgraph.node_attr['right'].get(max_score_edge[1]))
    
    # Formal Context
    # --------------
    fc = FormalContext(subgraph)
    print(f'Formal context dimensions: {fc.I.shape}')

    # Derivation operators: intention, extention
    a = fc.G[0:2]
    b = fc.M[0:2]
    print(f'Intention of objects {a}: {fc.intention(a)}')
    print(f'Extension of attributes {b}: {fc.extension(b)}')
    print(f'Extention(intention({a})): {fc.extension(fc.intention(a))}')

    
    # Concept Lattice
    # ---------------
    
    # A formal concept is a pair `(A, B)` of objects `A` and attributes `B`. Ojbects `A` are all the objects 
    # sharing attributes `B`. Attributes `B` are all the attributes describing objects `A`. In other words:
    #   * `A = extension(B)`
    #   * `B = intention(A)`

    # TODO: implement ConcepLattice class
    print()
    L = ConceptLattice.from_context(fc)



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
    


if __name__ == '__main__':
    
    # Run on GoodReads poetry data
    # ----------------------------
    # run()

    # Run on toy data
    # ---------------
    run_toy()
    