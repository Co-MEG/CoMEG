from collections import defaultdict, Counter
from contextlib import redirect_stdout
import numpy as np
import pickle
from scipy import sparse, special
from tqdm import tqdm

from sknetwork.data import load_netset
from sknetwork.utils import get_degrees

from line_profiler import LineProfiler

def mdl_graph(adjacency):
    n = max(1, adjacency.shape[0])
    m = max(1, adjacency.nnz)
    
    # nodes
    nodes_mdl = np.log2(n)
    
    # edges
    #degrees = adjacency.dot(np.ones(n))
    #max_degree = np.max(degrees)
    #edges_mdl = (n + 1) * np.log2(max_degree + 1) + np.sum([np.log2(special.comb(n, deg)) for deg in degrees])
    edges_mdl = np.log2(m)

    if n == 1:
        return 1
    else:
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

def intention(nodes, context):
    if len(nodes) == 0:
        return np.arange(0, context.shape[1])
    else:
        return context[nodes, :].indices
    
def extension(attributes, context_csc):
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

def graph_unexpectedness(adjacency, gen_complexities):
    n = adjacency.shape[0]
    complexity_desc_g = mdl_graph(adjacency)
    complexity_gen_g = np.mean(gen_complexities.get(n))
    return complexity_gen_g - complexity_desc_g

def attr_unexpectedness(biadjacency, attributes, degrees):
    complexity_gen_a = np.log2(special.comb(biadjacency.shape[1], len(attributes)))
    complexity_desc_a = 0
    for a in attributes:
        complexity_desc_a += np.log2(degrees[a])
    return complexity_gen_a - complexity_desc_a

def pattern_unexpectedness(adjacency, biadjacency, gen_complexities, attributes, degrees):
    u_g = graph_unexpectedness(adjacency, gen_complexities)
    u_a = attr_unexpectedness(biadjacency, attributes, degrees)
    return u_g + u_a

def init_inclose(context):
    extents, intents = [], []
    extents_init = np.arange(context.shape[0])
    intents_init = []
    extents.append(extents_init) # Initalize extents with all objects from context
    intents.append(intents_init) # Initialize intents with empty set attributes
    return extents, intents

def comeg(adjacency, context, context_csc, extents, intents, r=0, y=0, min_support=0, max_support=np.inf, degs=[], unexs_g=[], unexs_a=[], unexs=[], names_col=[], comp_gen_graph=None):
    """InClose algorithm using Unexpectedness + IsCannonical function. """
    
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
        unset_ext_j = extension([j], context_csc)
        ext_j = set(unset_ext_j)
        extents[r_new] = list(sorted(set(extents[r]).intersection(ext_j)))
        len_new_extent = len(extents[r_new])
        
        if (len_new_extent >= min_support) and (len_new_extent <= max_support):
                       
            new_intent = list(sorted(set(intents[r]).union(set([j]))))
            
            # Compute Unexpectedness on pattern (i.e on graph and attributes)
            # ------------------------------------------------------------------------------------------------------------
            print(f'  Extent size {len(extents[r_new])} - intent {new_intent}')
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
                if unex - unexs[ptr] >= 0:
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

                    intents[r_new] = new_intent 
                    len_new_intent = len(intents[r_new])

                    print(f'r:{r} rnew:{r_new}')
                    print(f' ISCANNO comparing unex={unex} and unexs[{ptr}]=={unexs[ptr]}')
                    if unex - unexs[ptr] >= 0 or r == 0:   
                        unexs.append(unex)
                        ptr += 1
                        print(f'  --> Enter recursion with Intent: {names_col[intents[r_new]]}...')
                        comeg(adjacency, context, context_csc, extents, intents, r=r_new, y=j+1, min_support=min_support, max_support=max_support, 
                                     degs=degs, unexs_g=unexs_g, unexs_a=unexs_a, unexs=unexs, names_col=names_col, comp_gen_graph=comp_gen_graph)
                
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


def run_comeg(adjacency, biadjacency, words, complexity_gen_graphs):
    # Initialization
    extents, intents = init_inclose(biadjacency)
    degs = get_degrees(biadjacency, transpose=True)
    global r_new
    r_new = 0
    global ptr 
    ptr = 0

    # Degree of attribute = # articles in which it appears
    freq_attribute = get_degrees(biadjacency.astype(bool), transpose=True)
    index = np.flatnonzero((freq_attribute <= 1000) & (freq_attribute >= 5))

    # Filter data with index
    biadjacency = biadjacency[:, index]
    words = words[index]
    freq_attribute = freq_attribute[index]

    # Order attributes according to their ascending degree
    # This allows to add first attributes that will generate bigger subgraphs
    sort_index = np.argsort(freq_attribute)
    sorted_degs = freq_attribute[sort_index]
    filt_biadjacency = biadjacency[:, sort_index]
    sorted_names_col = words[sort_index]

    # Convert context to csc at first, to fasten algorithm
    filt_biadjacency_csc = filt_biadjacency.tocsc()
    print(f'Context: {filt_biadjacency.shape}')

    # Algorithm
    with open('log_time_function', 'w') as f:
        with redirect_stdout(f):
            print('starts profiling...')
            lp = LineProfiler()
            lp_wrapper = lp(comeg)
            lp_wrapper(adjacency, filt_biadjacency, filt_biadjacency_csc, extents, intents, r=0, y=0, 
                                    min_support=5, max_support=15,
                                    degs=sorted_degs, unexs_g=[0], unexs_a=[0], unexs=[0], names_col=sorted_names_col,
                                    comp_gen_graph=complexity_gen_graphs)
            lp.print_stats()

    res = [*zip(extents, intents)]
    print(len(res))

    # Save result
    #with open("result_time_function.bin", "wb") as output:
    #    pickle.dump(concepts_iscanno, output)

    #print(len(concepts_iscanno))

graph = load_netset('wikischools')
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

for num_n in tqdm(range(1, 100)):
    for j in range(30):
        sel_attrs = np.random.choice(attrs_indexes, size=num_n, replace=False, p=attrs_degrees)
        sel_nodes = extension(sel_attrs, biadjacency_csc)
        sel_g = adjacency[sel_nodes, :][:, sel_nodes]
        mdl = mdl_graph(sel_g)
        #mdl = np.log2(len(sel_nodes)) # mdl is just the complexity of the number of nodes
        complexity_gen_graphs[num_n].append(mdl)

run_comeg(adjacency, biadjacency, words, complexity_gen_graphs)
