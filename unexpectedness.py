from collections import defaultdict, Counter
from contextlib import redirect_stdout
from line_profiler import LineProfiler
import numpy as np
import pickle
from scipy import sparse, special
import time
from tqdm import tqdm
from typing import List

from sknetwork.data import load_netset
from sknetwork.utils import get_degrees, get_neighbors


def mdl_graph(adjacency) -> float:
    """Minimum description length for graph structure.
    
    Parameters
    ----------
    adjacency: sparse.csr_matric
        Adjacency matrix of the graph
        
    Outputs
    -------
        Minimum description length of graph structure."""
    n = adjacency.shape[0]
    m = adjacency.nnz

    if n == 0:
        return 0
    else:
        # nodes
        nodes_mdl = np.log2(n + 1)
        
        # edges
        degrees = adjacency.dot(np.ones(n))
        max_degree = np.max(degrees)
        edges_mdl = (n + 1) * np.log2(max_degree + 1) + np.sum([np.log2(special.comb(n, deg)) for deg in degrees])
        #edges_mdl = np.log2(m+1)

        return nodes_mdl + edges_mdl
        
def is_cannonical(context, extents, intents, r, y):
    global r_new

    for k in range(len(intents[r])-1, -1, -1):
        for j in range(y, intents[r][k], -1):            
            for h in range(len(extents[r_new])):
                if context[extents[r_new][h], j] == 0:
                    h -= 1 # Necessary for next test in case last interaction of h for-loop returns False
                    break
            if h == len(extents[r_new]) - 1:
                return False
        y = intents[r][k] - 1

    for j in reversed(range(y, -1, -1)):
        for h in range(len(extents[r_new])):
            if context[extents[r_new][h], j] == 0:
                h -= 1 # Necessary for next test in case last interaction of h for-loop returns False
                break
        if h == len(extents[r_new]) - 1:
            return False
    
    return True

def intention(nodes, context) -> np.ndarray:
    """Intention of an array of nodes.
    
    Parameters
    ----------
    nodes: np.ndarray
        Array of node indexes
    context: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
        
    Outputs
    -------
        Array of attributes shared by all nodes."""
    if len(nodes) == 0:
        return np.arange(0, context.shape[1])
    intent = get_neighbors(context, node=nodes[0])
    if len(nodes) == 1:
        return intent
    else:
        intent = set(intent)
        for o in nodes[1:]:
            intent &= set(get_neighbors(context, node=o))
            if len(intent) == 0:
                break
        return np.array(list(intent))
    
def extension_csc(attributes, context_csc) -> np.ndarray:
    """Extension of an array of attributes, using CSC format.
    
    Parameters
    ----------
    attributes: np.ndarray
        Array of attribute indexes
    context_csc: sparse.csc_matrix
        Features matrix of the graph in csc format. Contains nodes x attributes.
        
    Outputs
    -------
        Array of nodes sharing all attributes."""
    if len(attributes) == 0:
        return np.arange(0, context_csc.shape[0])
    else:
        res = set(context_csc[:, attributes[0]].indices)
        if len(attributes) > 1:
            for a in attributes[1:]:
                res &= set(context_csc[:, a].indices)
                if len(res) == 0:
                    break
        return np.array(list(res))

def extension(attributes, context):
    """Extension of an array of attributes.
    
    Parameters
    ----------
    attributes: np.ndarray
        Array of attribute indexes
    context: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
        
    Outputs
    -------
        Array of nodes sharing all attributes."""
    ext = get_neighbors(context, node=attributes[0], transpose=True)
    if len(attributes) == 1:
        return ext
    else:
        ext = set(ext)
        for a in attributes[1:]:
            ext &= set(get_neighbors(context, node=a, transpose=True))
            if len(ext) == 0:
                break
        return np.array(list(ext))

def graph_unexpectedness(adjacency, gen_complexities) -> float:
    """Unexpectedness of a graph structure.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph.
    gen_complexities: dict
         Dictionnary with number of nodes as keys and list of graph generation complexities as values.
         
    Outputs
    -------
        Unexpectedness of a graph structure as a float value. """
    n = adjacency.shape[0]
    complexity_desc_g = mdl_graph(adjacency.astype(bool) + sparse.identity(n).astype(bool))
    complexity_gen_g = np.mean(gen_complexities.get(n))
    return complexity_gen_g - complexity_desc_g

def attr_unexpectedness(biadjacency, attributes, degrees) -> float:
    """Unexpectedness of a list of attributes.
    
    Parameters
    ----------
    biadjacency: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
    attributes: list
        List of attribute indexes.
    degrees: np.ndarray
        Array of attribute degrees in biadjacency
         
    Outputs
    -------
        Unexpectedness of list of attributes as a float value. """
    complexity_gen_a = np.log2(special.comb(biadjacency.shape[1], len(attributes)))
    complexity_desc_a = 0
    for a in attributes:
        complexity_desc_a += np.log2(degrees[a])
    return complexity_gen_a - complexity_desc_a

def pattern_unexpectedness(adjacency, biadjacency, gen_complexities, attributes, degrees):
    u_g = graph_unexpectedness(adjacency, gen_complexities)
    u_a = attr_unexpectedness(biadjacency, attributes, degrees)
    return u_g + u_a

def init_comeg(context) -> tuple:
    """Initialization for comeg algorithm.
    
    Parameters
    ---------
    context: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
        
    Returns
    -------
        Tuple of two lists, containing all nodes in graph and empty list of attributes. """
    extents, intents = [], []
    extents_init = np.arange(context.shape[0])
    intents_init = []
    extents.append(extents_init) # Initalize extents with all objects from context
    intents.append(intents_init) # Initialize intents with empty set attributes
    return extents, intents

def comeg(adjacency, context, context_csc, extents, intents, r=0, y=0, min_support=0, max_support=np.inf, beta=0, 
            degs=[], unexs_g=[], unexs_a=[], unexs=[], names_col=[], comp_gen_graph=None) -> List:
    """InClose algorithm using Unexpectedness + IsCannonical function. 
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    context: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
    context_csc: sparse.csc_matrix
        Features matrix of the graph in CSC format.
    extents: list
        List of extents, i.e sets of nodes.
    intents: list
        List of intents, i.e sets of attributes.
    r: int (default=0)
        Index of the pattern being filled.
    y: int (default=0)
        Index of candidate attribute.
    min_support: int (default=0)
        Minimum support value for extent.
    max_support: int (default +inf)
        Maximum support value for extent.
    beta: int (default=0)
        Minimum support value for intent.
    degs, unexs_g, unexs_a, unexs, names_col: list
        Lists for value storage over recursion.
    comp_gen_graph: dict (default=None)
        Dictionnary with number of nodes as keys and list of graph generation complexities as values.
        
    Returns
    -------
        List of tuples where each tuple is an unexpected pattern made of (extent, intent). 
    """
    global r_new
    global ptr
    r_new = r_new + 1
    
    print(f'NEW ITERATION \n --------')
    print(f'r: {r} - r_new: {r_new}')
    # ------------------------------------------------
    print(f'|extents[r]|: {len(extents[r])} - intents[r]: {names_col[intents[r]]}')
    
    for j in np.arange(context.shape[1])[y:]:
        try:
            extents[r_new] = []
            unexs_g[r_new] = 0
            unexs_a[r_new] = 0
        except IndexError:
            extents.append([])
            unexs_g.append(0)
            unexs_a.append(0)

        # Form a new extent by adding extension of attribute j to current concept extent
        ext_j = set(extension([j], context_csc))
        #ext_j = set(extension([j], context))
        extents[r_new] = list(sorted(set(extents[r]).intersection(ext_j)))
        len_new_extent = len(extents[r_new])
        
        if (len_new_extent >= min_support) and (len_new_extent <= max_support):

            # Verify that length of intention of new extent is greater than a threshold (e.g beta)
            # In other words, we only enter the loop if the new extent still has "space" to welcome enough new attributes
            # Using this, we can trim all patterns with not enough attributes from the recursion tree
            size_intention = len(intention(extents[r_new], context))
            if size_intention >= beta:
                    
                new_intent = list(sorted(set(intents[r]).union(set([j]))))
                
                # Compute Unexpectedness on pattern (i.e on graph and attributes)
                # ------------------------------------------------------------------------------------------------------------
                print(f'  Extent size {len(extents[r_new])} - intent {new_intent}')
                size = len(new_intent)
                unex_g = graph_unexpectedness(adjacency[extents[r_new], :][:, extents[r_new]], comp_gen_graph)
                unexs_g[r_new] = unex_g
                # Attributes unexpectedness
                unex_a = attr_unexpectedness(context, new_intent, degs)
                unexs_a[r_new] = unex_a
                # Total unexpectedness
                unex = unex_g + unex_a
                #unexs[r_new] = unex
                print(f'  U(G): {unex_g}')
                print(f'  U(A): {unex_a}')
                print(f'  U: {unex}')
                print(f'unexs: {unexs} r_new: {r_new} - r: {r} - ptr: {ptr}')
                # ------------------------------------------------------------------------------------------------------------
                
                if len_new_extent - len(extents[r]) == 0:
                    print(f' == comparing unex={unex} and unexs[{ptr}]={unexs[ptr]}')
                    #if unex - unexs[ptr] >= 0:
                    if unex - unexs[ptr] >= -np.inf:
                        print(f'  Extent size did not change -> attribute {names_col[j]} is added to intent.')
                        intents[r] = new_intent
                        unexs[-1] = unex
                    else:
                        print(f'STOP rec, unexpectedness difference is {unex - unexs[ptr]}')
                        print(f'Attribute {names_col[j]} ({j}) does not add any unexpectedness to pattern')
                        #extents[r_new].pop(-1) -> no need to change the extent since we are in the block where it did not move by adding attribute
                        #intents[r_new].pop(-1) -> at this stage, we only use new-intent, so no need to remove anything from intents parameter
                        #raise Exception('end')
                        break
                    
                else:
                    is_canno = is_cannonical(context, extents, intents, r, j - 1)
                    if is_canno:
                        print(f'extents {extents[r]} intents {intents[r]} r {r} rnew {r_new}')
                        print(f'  Extent size DID change. IsCannonical: {is_canno}')
                        try:
                            intents[r_new] = []
                        except IndexError:
                            intents.append([])

                        #intents[r_new] = new_intent 
                        #len_new_intent = len(intents[r_new])

                        print(f'r:{r} rnew:{r_new}')
                        print(f' ISCANNO comparing unex={unex} and unexs[{ptr}]=={unexs[ptr]}')
                        #if unex - unexs[ptr] >= 0 or r == 0:
                        if unex - unexs[ptr] >= -np.inf:
                            
                            intents[r_new] = new_intent 
                            len_new_intent = len(intents[r_new])

                            unexs.append(unex)
                            ptr += 1
                            print(f'  --> Enter recursion with Intent: {names_col[intents[r_new]]}...')
                            comeg(adjacency, context, context_csc, extents, intents, r=r_new, y=j+1, min_support=min_support, 
                                        max_support=max_support, beta=beta, degs=degs, unexs_g=unexs_g, 
                                        unexs_a=unexs_a, unexs=unexs, names_col=names_col, comp_gen_graph=comp_gen_graph)
                        else:
                            print(f'IsCANNO but no U improvement')
                            break
                    
                    else:
                        print(f'IsCannonical: False --> do not enter recursion.')
                    
    print(f'inexs: {unexs}')        
    print(f'r:{r} - r_new:{r_new}')
    unexs.pop(-1)
    ptr -= 1
    print(f'inexs after pop: {unexs}')        
    print(f'**END FUNCTION')
    #print(f'**concept: ({[*zip(extents, intents)]})')
    
    return [*zip(extents, intents)]


def run_comeg(adjacency, biadjacency, words, complexity_gen_graphs, order_attributes, s, beta, outfile):
    """Run concept mining algorithm.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph.
    biadjacency: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
    words: np.ndarray
        Features names.
    complexity_gen_graph: dict
        Dictionnary with number of nodes as keys and list of graph generation complexities as values.
    order_attributes: bool
        If True, order attributes according to their ascending degree.
    s: int
        Minimum extent support.
    beta: int
        Minimum intent support.
    outfile: str
        Output filename.
    """
    # Initialization
    extents, intents = init_comeg(biadjacency)
    degs = get_degrees(biadjacency, transpose=True)
    global r_new
    r_new = 0
    global ptr 
    ptr = 0

    # Degree of attribute = # articles in which it appears
    freq_attribute = get_degrees(biadjacency.astype(bool), transpose=True)
    index = np.flatnonzero((freq_attribute <= 1000) & (freq_attribute >= s))

    # Filter data with index
    biadjacency = biadjacency[:, index]
    words = words[index]
    freq_attribute = freq_attribute[index]

    # Order attributes according to their ascending degree
    # This allows to add first attributes that will generate bigger subgraphs
    if order_attributes:
        sort_index = np.argsort(freq_attribute)
    else:
        sort_index = np.arange(0, len(freq_attribute))
    sorted_degs = freq_attribute[sort_index]
    filt_biadjacency = biadjacency[:, sort_index]
    sorted_names_col = words[sort_index]

    # Convert context to csc at first, to fasten algorithm
    filt_biadjacency_csc = filt_biadjacency.tocsc()
    print(f'Context: {filt_biadjacency.shape}')

    # Algorithm
    with open(f'log_{outfile}', 'w') as f:
        with redirect_stdout(f):
            print('starts profiling...')
            lp = LineProfiler()
            lp_wrapper = lp(comeg)
            lp_wrapper(adjacency, filt_biadjacency, filt_biadjacency_csc, extents, intents, r=0, y=0, 
                                    min_support=s, max_support=100, beta=beta,
                                    degs=sorted_degs, unexs_g=[0], unexs_a=[0], unexs=[0], names_col=sorted_names_col,
                                    comp_gen_graph=complexity_gen_graphs)
            lp.print_stats()

    res = [*zip(extents, intents)]

    # Save result
    with open(f"result_{outfile}.bin", "wb") as output:
        pickle.dump(res, output)

    return len(res) 

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

