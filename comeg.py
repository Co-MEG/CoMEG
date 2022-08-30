import time
from src.algorithms import AdamicAdar
from src.concept_lattice import ConceptLattice
from src.context import FormalContext
from src.graph import BipartiteGraph

from sknetwork.utils import get_neighbors as gn
from sknetwork.linkpred import JaccardIndex #, AdamicAdar
import numpy as np
import json
import os
import random

from sklearn import metrics

import matplotlib.pyplot as plt

from concepts import Context

from IPython.display import SVG
from sknetwork.visualization import svg_graph, svg_bigraph
from sknetwork.utils import get_degrees

import networkx as nx

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


def run_toy_concept_lattice():
    # Toy context
    context_toy = FormalContext.from_csv(os.path.join(os.getcwd(), 'data/toy/context.csv'))
    print(context_toy.I.todense().shape)
    print(context_toy.G)
    print(context_toy.M)
    print()

    # Verif
    p = os.path.join(os.getcwd(), 'data/toy/context_concepts_frmt.csv')
    context_toy.to_concept_csv(p)
    c = Context.fromfile(p, frmat='csv')
    l = c.lattice
    for e, i in l:
        print(e, i)
    print(len(l))
    l_hashes = [] # list of sets
    for extent, intent in l:
        l_hashes.append(set(extent).union(set(intent)))

     # Concept Lattice
    #L = ConceptLattice.from_context(context_toy, algo='in-close')
    lattice = context_toy.lattice(algo='in-close')
    print()
    for c in lattice.concepts:
        if set(c[0]).union(set(c[1])) in l_hashes:
            print('* ', c)
        else:
            print(f'X ', c)
    print(f'Total number of concepts: {len(lattice)}')



def run_prediction():
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
    if TRAIN_TEST_SPLIT:
        print('Splitting graph into train-test...')
        train_g, test_g = g.train_test_split(test_size=TEST_SIZE)
        print(f'Number of reviews in test: {len(test_g.E)}')
        print(f'Number of left nodes used: {len(set([e[0] for e in test_g.E]))}')
        print(f'Number of right nodes used: {len(set([e[1] for e in test_g.E]))}')

    # Adamic Adar
    print('Adamic Adar index...')
    aa = AdamicAdar()

    # predit random edges
    rand_n_left = random.choices(train_g.V.get('left'), k=len(test_g.E))
    rand_n_rigth = random.choices(train_g.V.get('right'), k=len(test_g.E))
    rand_edges = [(u, v) for u, v in zip(rand_n_left, rand_n_rigth)]
    rand_scores =  aa.predict_edges(train_g, rand_edges, transpose=False)

    # predict edges
    scores = aa.predict_edges(train_g, test_g.E, transpose=False)
    print(f'Number of predictions: {len(scores)}')
    
    # Save results for AA index
    #res = os.path.join(PATH_RES, 'adamic_adar_test_graph.json')
    #with open(res, "w") as f:
    #    json.dump(scores , f) 

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
    

def load_predictions(path):
    # Load predictions
    D = {}
    # Convert list of dicts into one big dictionary of values
    print(f'Load result values...')
    with open(path) as f:
        lines = f.read().splitlines()
        for l in lines:            
            pair = l.strip('{}').split(': ')
            k = pair[0]
            v = pair[1]
            vals = []
            for idx, i in enumerate(v.strip('[]').split(', ')):
                if idx <= 1:
                    vals.append(i.strip('""'))
                else:
                    vals.append(float(i))
            D.update({k: vals})

    return D

def run_auc_test_graph():

    # Build graph
    g = BipartiteGraph()
    g.load_data(IN_PATH, use_cache=USE_CACHE)

    # Load predictions
    path = os.path.join(PATH_RES, 'adamic_adar_test_graph.json')
    preds_test = load_predictions(path)
    path_rand = os.path.join(PATH_RES, 'adamic_adar_test_graph_rand.json')
    preds_rand = load_predictions(path_rand)

    # ground truth label
    y_pred_test, y_true_test = zip(*[(values[2], g.has_edge(values[0], values[1])) for _, values in preds_test.items()])
    y_pred_rand, y_true_rand = zip(*[(values[2], g.has_edge(values[0], values[1])) for _, values in preds_rand.items()])
    
    y_pred = y_pred_test + y_pred_rand
    y_true = y_true_test + y_true_rand
    print(f'Number of predicted edges: {len(y_pred)}')
    
    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    plot_roc_auc(y_true, y_pred, ax)
    #plt.show()
    plt.savefig(os.path.join(PATH_RES, 'img', 'adamic_adar_test_graph.eps'))


def run_concept_lattice():
    # Build graph
    g = BipartiteGraph()
    g.load_data(IN_PATH, use_cache=USE_CACHE)

    # Load predictions
    path = os.path.join(PATH_RES, 'adamic_adar_test_graph.json')
    preds = load_predictions(path)

    # Formal Concept Analysis
    # -----------------------
    # Subgraph in the vicinity of an edge
    pred_scores, pred_edges = zip(*[(values[2], (values[0], values[1])) for _, values in preds.items()])
    
    # Max score edge is for edge ('fd379cf294fc1937e41f3f7df3c9eabe', '1381'):
    #   book '1381' is "The Odyssey (Homere)"
    #score_edge = pred_edges[np.argmax(pred_scores)]
    #print(f'Predicted edge: {score_edge}')

    # Random score edge
    i = np.random.randint(len(pred_scores))
    score_edge = pred_edges[i]
    
    # Specific edge
    #score_edge = ('fd379cf294fc1937e41f3f7df3c9eabe', '1381') # max score edge
    #score_edge = ('bc1d727746e210f315138932e0aacb11', '13637887') # small context
    score_edge = ('c36d77d30d627e8ad5eccbab8d92f54d', '22237148') # medium context
    #score_edge = ('18bf7556e8f06efd9269db97880dd9ef', '5289') # Medium context (20, 92)
    #score_edge = ('0f9695b816f80ed7dd2768beaab2fabc', '8744427') # Large context (173, 491)
    #score_edge = ('9ec2bf0c6452ee5be914cb3330e233e6', '25052044') # Large context (225, 718) - InClose -> 40s

    print(f'Predicted edge: {score_edge}')

    subgraph = g.subgraph_vicinity(score_edge)
    print(f"# nodes in subgraph: {len(subgraph.V['left'])}, {len(subgraph.V['right'])}")
    print(f'# edges in subgraph: {len(subgraph.E)}')
    print(f"# of edge attributes: {len(subgraph.edge_attr)}")
    print(f"# of node attributes right: {len(subgraph.node_attr['right'])}")
    print(f'Edge exists in graph: {g.has_edge(u=score_edge[0], v=score_edge[1])}')
    
    # Formal Context
    # --------------
    fc = FormalContext(subgraph)
    print(f'Formal context dimensions: {fc.I.shape}')

    fc_path = os.path.join(PATH_RES, 'context', f'context_{score_edge[0]}_{score_edge[1]}.csv')
    #fc.to_csv(fc_path, sep=',')
    fc.to_concept_csv(fc_path)

    # Derivation operators: intention, extention
    #a = fc.G[0:2]
    #b = fc.M[0:2]
    #print(f'Intention of objects {a}: {fc.intention(a)}')
    #print(f'Extension of attributes {b}: {fc.extension(b)}')
    #print(f'Extention(intention({a})): {fc.extension(fc.intention(a))}')

    # Concept Lattice
    # ---------------
    
    # A formal concept is a pair `(A, B)` of objects `A` and attributes `B`. Ojbects `A` are all the objects 
    # sharing attributes `B`. Attributes `B` are all the attributes describing objects `A`. In other words:
    #   * `A = extension(B)`
    #   * `B = intention(A)`
    algo = 'in-close'
    start = time.time()
    lattice = fc.lattice(algo=algo)
    print(f'Elapsed time using {algo}: {time.time()-start}')
    print(f'Number of concepts in lattice: {len(lattice)}')

    """algo = 'CbO'
    start = time.time()
    lattice_cbo = fc.lattice(algo=algo)
    print(f'Elapsed time using {algo}: {time.time()-start}')
    print(f'Number of concepts in lattice: {len(lattice_cbo)}')"""

    # Verify list of concepts with Concepts library
    c = Context.fromfile(fc_path, frmat='csv')
    l = c.lattice
    print(f'Number of concepts in concepts lattice: {len(l)}\n')
    l_hashes = [] # list of sets
    for extent, intent in l:
        l_hashes.append(set(extent).union(set(intent)))
    """print(f'Verify CbO algorithm')
    print(f'===========================================================')
    for c in lattice_cbo.concepts:
        if set(c[0]).union(set(c[1])) in l_hashes:
            print(f'* Concept exists in other result {c[0]}')
        else:
            print(f'X Concept DOES NOT exists in other result {c[0]} ')"""
    """print(f'Verify InClose algorithm')
    print(f'===========================================================')
    for c in lattice.concepts:
        if set(c[0]).union(set(c[1])) in l_hashes:
            print(f'* Concept exists in other result {c[0]}')
        else:
            print(f'X Concept DOES NOT exists in other result {c[0]} ')"""

    # Print all concepts in lattices
    # ======================================
    """print(f'\nLattice CbO')
    print(f'-------------------------------')
    for c in lattice_cbo.concepts:
        print(c)

    print(f'\nLattice Concepts')
    print(f'-------------------------------')
    for e, i in l:
        print(e, i)

    print(f'\nLattice inclose')
    print(f'-------------------------------')
    for c in lattice.concepts:
        print(c)"""
    
    topk_concepts = lattice.top_k(metric='MILP')
    print(f'\n ****** Selected concepts: ')
    for c in topk_concepts:
        print(c)
    
    query_ext = {'degree_left': 2, 'degree_right': 1}
    query_int = ['country_code_US', 'language_code_eng']
    filtered_concepts = lattice.filter(query_ext, query_int)
    print(f'\n ****** Filtered concepts: ')
    for c in filtered_concepts:
        print(c)


    G = nx.Graph()
    G.add_nodes_from(subgraph.V['left'], bipartite=0)
    G.add_nodes_from(subgraph.V['right'], bipartite=1)
    for e in subgraph.E:
        if e == score_edge:
            G.add_edge(e[0], e[1], color='r', weight=2)
        else:
            G.add_edge(e[0], e[1], color='black', weight=1)

    fig, axs = plt.subplots(3, 3, figsize=(30, 15))
    nx.draw_networkx(
                G,
                pos = nx.drawing.layout.bipartite_layout(G, nx.bipartite.sets(G)[0]),
                edge_color = [G[u][v]['color'] for u, v in G.edges()],
                width = [G[u][v]['weight'] for u, v in G.edges()],
                ax = axs.ravel()[0]
        )

    def render_title(title):
        res = ''
        i = 0
        for elem in title:
            if i >= 2:
                res += elem + ', '
            else:
                res += elem + ', \n'
            i += 1
        return res
        
    for c, ax in zip(filtered_concepts, axs.ravel()[1:]):
        # Build graph induced by concept        
        G_sub = G.copy()
        r_nodes_to_remove = list(set(subgraph.V['right']).difference(set(c[0])))
        G_sub.remove_nodes_from(r_nodes_to_remove)
        l_nodes_to_remove = list(set(subgraph.V['left']).difference(set([e[0] for e in G_sub.edges()])))
        G_sub.remove_nodes_from(l_nodes_to_remove)
        # Draw graph
        nx.draw_networkx(
                G_sub,
                pos = nx.drawing.layout.bipartite_layout(G_sub, nx.bipartite.sets(G_sub)[0]),
                edge_color = [G[u][v]['color'] for u, v in G.edges()],
                width = [G[u][v]['weight'] for u, v in G.edges()],
                ax = ax
        )
        ax.set_title(f'{render_title(c[1])}', fontsize=6)
    plt.show()

    """for c in filtered_concepts:
        col_idxs = [lattice.context.G2idx.get(i) for i in c[0]]
        g_concept = lattice.context.graph.adjacency_csr.T[col_idxs].T
        row_idxs = np.flatnonzero(get_degrees(g_concept))
        g_concept = g_concept[row_idxs, :]
        g_coo = g_concept.to_coo()

        G = nx.graph()
        G.add_nodes_from(lattice.context.G[row_idxs], bipartite=0)
        G.add_nodes_from(lattice.context.M[col_idxs], bipartite=1)
        G.add_edges_from()"""


    """mat = lattice.pairwise_concept_distance()
    fig, ax = plt.subplots()
    ax.matshow(mat, cmap=plt.cm.Blues)
    tick_marks = np.arange(len(lattice.concepts))
    plt.xticks(tick_marks, range(len(lattice.concepts)), fontsize=7)
    plt.yticks(tick_marks, [x[0] for x in lattice.concepts], fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            c = round(mat[j, i], 2)
            ax.text(i, j, str(c), va='center', ha='center')
    #plt.title(f'Pairwise concept distance (Jaccard) for {score_edge}', weight='bold')
    
    plt.show()
    res = os.path.join(PATH_RES, 'img', f'top_5_concepts_{score_edge[0]}_{score_edge[1]}.eps')
    plt.tight_layout()
    plt.savefig(res)"""

if __name__ == '__main__':
    
    # Run on GoodReads poetry data
    # ----------------------------
    #run_prediction()

    # Evaluation (AUC) on test graph
    # ------------------------------
    #run_auc_test_graph()

    # Concept lattice
    # ---------------
    # Toy data
    #run_toy_concept_lattice()
    
    # Full data
    run_concept_lattice()
