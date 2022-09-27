from concepts import Context
from IPython.display import SVG
import matplotlib.pyplot as plt
import networkx as nx
from nltk.tokenize import word_tokenize
import numpy as np
import os
import random
from sklearn import metrics
import time
import warnings
warnings.filterwarnings("ignore")

from sknetwork.utils import get_neighbors as gn
from sknetwork.linkpred import JaccardIndex #, AdamicAdar
from sknetwork.visualization import svg_graph, svg_bigraph
from sknetwork.utils import get_degrees

from src.algorithms import AdamicAdar
from src.concept_lattice import ConceptLattice
from src.context import FormalContext
from src.graph import BipartiteGraph
from src.metric import cosine_similarity, jaccard_score
from src.utils import *

import nltk
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA

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


def run_concept_lattice(path: str, verify: str = True):

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
    i = np.random.randint(int(len(pred_scores) / 2))
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
    #score_edge = ('7b3befb2bef070b8c04e9865270fd0c1', '23356133') # Spiderman
    #score_edge = ('e9e8653b26d73f4c94f1b5e1a59146a2', '12962439') # GoT vol.1
    #score_edge = ('2bebb8e83292c4e65fb970eb953fbec5', '26109143') # Everything Is Teeth
    score_edge = ('3c057894279fd958980fbe7777c3942c', '25167353')

    print(f"Predicted edge: {score_edge} (random index: {i})")
    print(f"Title: {g.node_attr['right'].get(score_edge[1]).get('title')}")
    print(f"Description: {g.node_attr['right'].get(score_edge[1]).get('description')}")

    subgraph = g.subgraph_vicinity(score_edge, method='ppr')
    #subgraph = g.subgraph_vicinity(score_edge)
    #subgraph = g.subgraph_vicinity_degree(score_edge, min_degree_left=10, min_degree_right=15) # Larger vicinity
    print(subgraph.adjacency_csr.shape, subgraph.adjacency_csr.nnz)


    # Smaller vicinity
    #subgraph2 = g.subgraph_vicinity(score_edge)
    #print(subgraph2.adjacency_csr.shape)

    print(f"# nodes in subgraph: {len(subgraph.V['left'])}, {len(subgraph.V['right'])}")
    print(f'# edges in subgraph: {len(subgraph.E)}')
    print(f"# of edge attributes: {len(subgraph.edge_attr)}")
    print(f"# of node attributes right: {len(subgraph.node_attr['right'])}")
    print(f'Edge exists in graph: {g.has_edge(u=score_edge[0], v=score_edge[1])}')
    
    global edge_exist
    edge_exist = g.has_edge(u=score_edge[0], v=score_edge[1])
    
    if len(subgraph.V['left']) > 5000:
        raise Exception('Number of left node is too large')

    # Formal Context
    # --------------
    fc = FormalContext(subgraph, use_description=True)
    print(f'Formal context dimensions: {fc.I.shape}')

    print(f'Filtering context using tf-idf...')
    fc.filter_transform(method='tf-idf', k=200)
    print(f'Filtered context using tf-idf: {fc.I.shape}')
    

    # Concept Lattice
    # ---------------
    
    # A formal concept is a pair `(A, B)` of objects `A` and attributes `B`. Ojbects `A` are all the objects 
    # sharing attributes `B`. Attributes `B` are all the attributes describing objects `A`. In other words:
    #   * `A = extension(B)`
    #   * `B = intention(A)`
    algo = 'in-close'
    start = time.time()
    lattice = fc.lattice(algo=algo, minimum_support = 0, maximum_support = 1000)
    print(f'Elapsed time using {algo}: {time.time()-start}')
    print(f'Number of concepts in lattice: {len(lattice)}')

    if verify:
        fc_path = os.path.join(PATH_RES, 'context', f'context_{score_edge[0]}_{score_edge[1]}.csv')
        fc.to_concept_csv(fc_path)

        # Verify list of concepts with Concepts library
        c = Context.fromfile(fc_path, frmat='csv')
        l = c.lattice
        print(f'Number of concepts in concepts lattice: {len(l)}\n')
        l_hashes = [] # list of sets
        for extent, intent in l:
            l_hashes.append(set(extent).union(set(intent)))
        print(f'Verify InClose algorithm')
        print(f'===========================================================')
        not_ex = 0
        ex = 0
        for c in lattice.concepts:
            if set(c[0]).union(set(c[1])) in l_hashes:
                #print(f'* Concept exists in other result {c[0]}')
                ex += 1
            else:
                #print(f'X Concept DOES NOT exists in other result {c[0]} ')
                not_ex += 1
        print(f'Number of existing concepts:     {(ex)}/{len(lattice.concepts)}')
        print(f'Number of non existing concepts: {(not_ex)}/{len(lattice.concepts)}')


    # FILTER CONCEPTS
    """topk_concepts_jaccard = lattice.top_k(metric='Jaccard')
    print(f'\n ****** Selected concepts using Jaccard: ')
    for c in topk_concepts_jaccard:
        print(c)

    plot_subgraphs(score_edge, subgraph, topk_concepts_jaccard, title=f'topk_jaccard_{score_edge}')"""
    
    all_tfidf_concepts = lattice.top_k(k=100, metric='tf-idf')
    print(f'\n ****** All concepts using TFIDF: ')
    for c in all_tfidf_concepts:
        print(c)
    topk_tfidf_concepts = lattice.top_k(k=10, metric='tf-idf')
    print(f'\n ****** Selected concepts using TFIDF: ')
    for c in topk_tfidf_concepts:
        print(c)
    print(f'\n ****** Selected concepts using TFIDF with pred book: ')
    sel = []
    for c in topk_tfidf_concepts:
        if score_edge[1] in c[0]:
            sel.append(c)
            print(c)

    plot_subgraphs(score_edge, subgraph, topk_tfidf_concepts, title=f'topk_TFIDF_{score_edge}', use_description=True)

    topk_concepts = lattice.top_k(metric='MILP')
    """print(f'\n ****** Selected concepts using MILP: ')
    sel = []
    for c in topk_concepts:
        print(c)
            
    for c in lattice.concepts:
        if score_edge[1] in c[0]:
            if c in topk_concepts:
                print(f'|extent|: {len(c[0])} - |intent|: {len(c[1])} **')
            else:
                if len(c[0]) < 10:
                    print(f'|extent|: {len(c[0])} - |intent|: {len(c[1])} - {c}')
                else:
                    print(f'|extent|: {len(c[0])} - |intent|: {len(c[1])}')"""

    plot_subgraphs(score_edge, subgraph, topk_concepts, title=f'topk_MILP_{score_edge}', use_description=True)
    #plot_subgraphs(score_edge, subgraph, sel, title=f'topk_MILP_{score_edge}', use_description=True)
    
    # User query
    # ----------
    query_ext = {'degree_left': 1, 'degree_right': 1, 'node_right': [score_edge[1]]}
    query_int = []
    # ----------

    filtered_concepts = lattice.filter(query_ext, query_int)
    """print(f'\n ****** Filtered concepts: ')
    for c in filtered_concepts:
        print(c)"""

    plot_subgraphs(score_edge, subgraph, filtered_concepts, title=f'filt_subgraphs_{score_edge}', use_description=True)


    # Evaluate explanations
    # ----------------------

    # we compute average-Jaccard between sets of attributes of predicted book and top-k sets of attributes derives from concepts intents
    # we compare average Jaccard scores with ExplaiNE method

    print()
    predicted_book_idx = fc.G2idx.get(score_edge[1])
    predicted_book_attrs = fc.M[fc.I.indices[fc.I.indptr[predicted_book_idx]:fc.I.indptr[predicted_book_idx+1]]]
    print(f'Attributes of predicted book (after tf-idf): {predicted_book_attrs}')
    
    my_top_concepts = all_tfidf_concepts #topk_tfidf_concepts

    j_score = 0
    j_concepts = []
    j_scores = []
    for c in my_top_concepts:
        #print(f'Score with {c[1]} = {jaccard_score(predicted_book_attrs, c[1]):.4f} ')
        j_score += jaccard_score(predicted_book_attrs, c[1])
        j_concepts.append(c[1])
        j_scores.append(jaccard_score(predicted_book_attrs, c[1]))

    print(f'\n****Top selected concepts using Jaccard with predicted node')
    mask = np.argsort(-np.asarray(j_scores))
    for idx, c in enumerate(np.asarray(j_concepts)[mask][:10]):
        print(f'Jaccard with {c} = {j_scores[mask[idx]]:.3f}')


    j_score = j_score / len(my_top_concepts)
    print(f'Avg J score: {j_score:.4f}')
    print()

    # word2vec embedding
    # we use word2vec embeddings of concept attributes and predicted node attribute to verify how close each
    # concept is compared to the predicted node. Indeed, using only Jaccard metric between sets of attributes
    # does not allow to consider semantically similar sets !
    
    corpus = []
    for c in my_top_concepts:
        if len(c[1]) > 0:
            corpus.append(c[1])

    # Word2Vec using CBOW
    model = Word2Vec(sentences=corpus, min_count=1, window=2)

    # average of word vectors for each set of attributes: we compute the barycenter of each set and compare them between each other
    predicted_bary = np.mean([model.wv[w] for w in predicted_book_attrs], axis=0)

    result_concepts = []
    cos_similarities = []
    all_barys = predicted_bary.copy()
    result_concepts.append(predicted_book_attrs)
    cos_similarities.append(round(cosine_similarity(predicted_bary, predicted_bary), 2))
    for c in my_top_concepts:
        # DO NOT take into account the predicted book itself
        if set(c[1]) != set(predicted_book_attrs) and len(c[1]) > 0:
            concept_bary = np.mean([model.wv[w] for w in c[1]], axis=0)
            all_barys = np.vstack((all_barys, concept_bary))
            cos_similarities.append(round(cosine_similarity(predicted_bary, concept_bary), 2))
            result_concepts.append(c[1])
    
    print('\n****Selected concepts using cosine similarity with word2vec')
    mask = np.argsort(-np.asarray(cos_similarities))
    for idx, c in enumerate(np.asarray(result_concepts)[mask][:10]):
        print(f'Cos sim with word2vec barycenters with {c} = {cos_similarities[mask[idx]]:.3f}')
    
    vectors = np.asarray(model.wv.vectors)
    vectors_labels = np.asarray(model.wv.index_to_key)

    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    tsne = TSNE(random_state=42, perplexity=5).fit_transform(all_barys)
    #pca = PCA(n_components=4)
    #pca_res = pca.fit_transform(vectors)
    #print(f'PCA explained variance ratio: {pca.explained_variance_ratio_}')
    num_concepts = len(corpus)
    palette = np.array(sns.color_palette('hls', num_concepts))
    #ax.scatter(tsne[:, 0], tsne[:, 1])
    for i in range(len(result_concepts)):
        ax.scatter(tsne[i, 0], tsne[i, 1], color=palette[i], label=result_concepts[i])
        if i == 0:
            ax.text(tsne[i, 0] + 0.01, tsne[i, 1], cos_similarities[i], color='red')
        else:
            ax.text(tsne[i, 0] + 0.01, tsne[i, 1], cos_similarities[i], color='black')
    #plt.legend(fontsize=5)
    #plt.show()




def plot_subgraphs(pred_edge, subgraph, concepts, title, use_description):  

    fig, axs = plt.subplots(3, 3, figsize=(30, 15))

    # Suptitle
    if not use_description:
        excluded_attributes = ['popular_shelves', 'description', 'link', 'url', 'image_url', \
                                                'book_id', 'isbn13', 'isbn', 'work_id', 'text_reviews_count', 'asin', \
                                                'kindle_asin', 'average_rating', 'ratings_count', 'num_pages', \
                                                'publication_day', 'similar_books', 'authors']
        suptitle = [str(k)+'_'+str(v) for k, v in subgraph.node_attr['right'].get(pred_edge[1]).items() if k not in excluded_attributes]
    else:
        suptitle = np.unique([x for x in word_tokenize(subgraph.node_attr['right'].get(pred_edge[1]).get('description')) if len(x) > 2])
    plt.suptitle(render_title(suptitle, k=10) + f' {edge_exist}', weight='bold')

    # Subgraph in the vicinity of the predicted edge
    G = nx.Graph()
    G.add_nodes_from(subgraph.V['left'], bipartite=0)
    G.add_nodes_from(subgraph.V['right'], bipartite=1)
    for e in subgraph.E:
        if e == pred_edge:
            G.add_edge(e[0], e[1], color='r', weight=2)
        else:
            G.add_edge(e[0], e[1], color='black', weight=1)
    
    # Color nodes
    color_map = []
    for n in G:
        if n == pred_edge[0] or n == pred_edge[1]:
            color_map.append('red')
        else:
            color_map.append('lightblue')

    draw_bipartite_graph(G, subgraph.V['left'], color_map, axs.ravel()[0])

    # Subgraphs induced by concepts
    for c, ax in zip(concepts[:8], axs.ravel()[1:]):
        # Build graph
        G_sub = G.copy()
        r_nodes_to_remove = list(set(subgraph.V['right']).difference(set(c[0])))
        G_sub.remove_nodes_from(r_nodes_to_remove)
        l_nodes_to_remove = list(set(subgraph.V['left']).difference(set([e[0] for e in G_sub.edges()])))
        G_sub.remove_nodes_from(l_nodes_to_remove)

        # Color nodes
        color_map_sub = []
        for n in G_sub:
            if n == pred_edge[0] or n == pred_edge[1]:
                color_map_sub.append('red')
            else:
                color_map_sub.append('lightblue')
        
        draw_bipartite_graph(G_sub, subgraph.V['left'], color_map_sub, ax)
        ax.set_title(f'{render_title(c[1])}', fontsize=10)
    
    title = title + '.eps'
    res = os.path.join(PATH_RES, 'img', title)
    plt.savefig(res)


# Data path
IN_PATH = 'goodreads_comics'  #'goodreads_poetry' 
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
    run_concept_lattice(f'{LINK_PRED_METHOD}_test_graph.json', verify=False)
