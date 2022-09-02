import time
import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt
from IPython.display import SVG

from src.algorithms import AdamicAdar
from src.concept_lattice import ConceptLattice
from src.context import FormalContext
from src.graph import BipartiteGraph
from src.utils import *

from sknetwork.utils import get_neighbors as gn
from sknetwork.linkpred import JaccardIndex #, AdamicAdar
from sknetwork.visualization import svg_graph, svg_bigraph
from sknetwork.utils import get_degrees

import networkx as nx

from sklearn import metrics

from concepts import Context


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


def run_concept_lattice(path: str):

    # Build graph
    g = BipartiteGraph()
    g.load_data(IN_PATH, use_cache=USE_CACHE)

    # Load predictions
    path = os.path.join(PATH_RES, path)
    preds = load_predictions(path)

    # Formal Concept Analysis
    # -----------------------
    # Subgraph in the vicinity of an edge
    pred_scores, pred_edges = zip(*[(values[2], (values[0], values[1])) for _, values in preds.items()])

    # Random score edge
    i = np.random.randint(len(pred_scores))
    score_edge = pred_edges[i]
    
    # Specific edge    
    # Poetry
    #score_edge = ('fd379cf294fc1937e41f3f7df3c9eabe', '1381') # max score edge
    #score_edge = ('bc1d727746e210f315138932e0aacb11', '13637887') # small context
    #score_edge = ('c36d77d30d627e8ad5eccbab8d92f54d', '22237148') # medium context
    #score_edge = ('18bf7556e8f06efd9269db97880dd9ef', '5289') # Medium context (20, 92)
    #score_edge = ('0f9695b816f80ed7dd2768beaab2fabc', '8744427') # Large context (173, 491)
    #score_edge = ('9ec2bf0c6452ee5be914cb3330e233e6', '25052044') # Large context (225, 718) - InClose: 40s
    #score_edge = ('c05512c006dd9ccb49b147ce619621d5', '11737010') # XL context (566, 1450) - InClose: 257s
    #score_edge = ('f2bac05b3932fe7c68960041744e5058', '489000') # XXL context (2332, 5375) -> ?
    # Comics
    #score_edge = ('2fea1cb2a1a8ef86c3dd751493d81b6b', '23310817')
    #score_edge = ('d2448cb7ff0e7a7a62875d6cb9a8abed', '9648722')

    print(f'Predicted edge: {score_edge}')

    # Larger vicinity
    subgraph = g.subgraph_vicinity_degree(score_edge)
    print(subgraph.adjacency_csr.shape)
    
    # Smaller vicinity
    #subgraph2 = g.subgraph_vicinity(score_edge)
    #print(subgraph2.adjacency_csr.shape)

    print(f"# nodes in subgraph: {len(subgraph.V['left'])}, {len(subgraph.V['right'])}")
    print(f'# edges in subgraph: {len(subgraph.E)}')
    print(f"# of edge attributes: {len(subgraph.edge_attr)}")
    print(f"# of node attributes right: {len(subgraph.node_attr['right'])}")
    print(f'Edge exists in graph: {g.has_edge(u=score_edge[0], v=score_edge[1])}')
    if len(subgraph.V['left']) > 1000:
        raise Exception('Number of left node is too large')

    # Formal Context
    # --------------
    fc = FormalContext(subgraph)
    print(f'Formal context dimensions: {fc.I.shape}')

    fc_path = os.path.join(PATH_RES, 'context', f'context_{score_edge[0]}_{score_edge[1]}.csv')
    #fc.to_csv(fc_path, sep=',')
    fc.to_concept_csv(fc_path)

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

    """topk_concepts_jaccard = lattice.top_k(metric='Jaccard')
    print(f'\n ****** Selected concepts using Jaccard: ')
    for c in topk_concepts_jaccard:
        print(c)

    plot_subgraphs(score_edge, subgraph, topk_concepts_jaccard, title=f'topk_jaccard_{score_edge}')"""
    
    
    topk_concepts = lattice.top_k(metric='MILP')
    print(f'\n ****** Selected concepts using MILP: ')
    sel = []
    for c in topk_concepts:
        if score_edge[1] in c[0]:
            print(c)
            sel.append(c)

    #plot_subgraphs(score_edge, subgraph, topk_concepts, title=f'topk_MILP_{score_edge}')
    plot_subgraphs(score_edge, subgraph, sel, title=f'topk_MILP_{score_edge}')
    
    # User query
    # ----------
    query_ext = {'degree_left': 1, 'degree_right': 1, 'node_right': [score_edge[1]]}
    query_int = []
    # ----------

    filtered_concepts = lattice.filter(query_ext, query_int)
    print(f'\n ****** Filtered concepts: ')
    for c in filtered_concepts:
        print(c)

    plot_subgraphs(score_edge, subgraph, filtered_concepts, title=f'filt_subgraphs_{score_edge}')


def plot_subgraphs(pred_edge, subgraph, concepts, title):
    
    excluded_attributes = ['popular_shelves', 'description', 'link', 'url', 'image_url', \
                                                'book_id', 'isbn13', 'isbn', 'work_id', 'text_reviews_count', 'asin', \
                                                'kindle_asin', 'average_rating', 'ratings_count', 'num_pages', \
                                                'publication_day', 'similar_books', 'authors']
    

    fig, axs = plt.subplots(3, 3, figsize=(30, 15))

    suptitle = [str(k)+'_'+str(v) for k, v in subgraph.node_attr['right'].get(pred_edge[1]).items() if k not in excluded_attributes]
    plt.suptitle(render_title(suptitle), weight='bold')

    # Subgraph in the vicinity of the predicted edge
    G = nx.Graph()
    G.add_nodes_from(subgraph.V['left'], bipartite=0)
    G.add_nodes_from(subgraph.V['right'], bipartite=1)
    for e in subgraph.E:
        if e == pred_edge:
            G.add_edge(e[0], e[1], color='r', weight=2)
        else:
            G.add_edge(e[0], e[1], color='black', weight=1)

    draw_bipartite_graph(G, subgraph.V['left'], axs.ravel()[0])

    # Subgraphs induced by concepts
    for c, ax in zip(concepts[:8], axs.ravel()[1:]):
        # Build graph
        G_sub = G.copy()
        r_nodes_to_remove = list(set(subgraph.V['right']).difference(set(c[0])))
        G_sub.remove_nodes_from(r_nodes_to_remove)
        l_nodes_to_remove = list(set(subgraph.V['left']).difference(set([e[0] for e in G_sub.edges()])))
        G_sub.remove_nodes_from(l_nodes_to_remove)
        
        draw_bipartite_graph(G_sub, subgraph.V['left'], ax)
        ax.set_title(f'{render_title(c[1])}', fontsize=10)
    
    title = title + '.eps'
    res = os.path.join(PATH_RES, 'img', title)
    plt.savefig(res)


# Data path
IN_PATH = 'goodreads_comics' #'goodreads_poetry'
USE_CACHE = True
TEST_SIZE = 0.05
TRAIN_TEST_SPLIT = True
PATH_RES = os.path.join(os.getcwd(), 'data', IN_PATH, 'result')
LINK_PRED_METHOD = 'spectral_emb' # 'adamic_adar'


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
    #run_concept_lattice('adamic_adar_test_graph.json')
    run_concept_lattice(f'{LINK_PRED_METHOD}_test_graph.json')
