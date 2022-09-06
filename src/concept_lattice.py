from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from src.concept import FormalConcept
if TYPE_CHECKING:
    from src.context import FormalContext
from src.metric import get_metric
from src.solver import Solver

from sknetwork.utils import get_degrees


class ConceptLattice:
    def __init__(self, concepts: List[FormalConcept]):
        # Build a Concept Lattice given formal concepts
        self.concepts = concepts
        self.context = None

    def __len__(self) -> int:
        """Return number of concepts in Lattice.

        Returns
        -------
        int
            Number of concepts
        """        
        return len(self.concepts)

    @classmethod
    def from_context(cls, context: FormalContext, algo: str = 'in-close', **kwargs) -> ConceptLattice:
        """Return a ``ConceptLattice`` according to ``context``, created using algorithm ``algo``.

        Parameters
        ----------
        context : FormalContext
            Formal context on which concept lattice is created.
        algo : str, optional
            Algorithm name, by default 'in-close' (in-close). Can be either:
                * 'in-close': In Close 
                * 'CbO': Close by One

        Returns
        -------
        ConceptLattice
            Concept Lattice
        """        
        algo_func = get_algo(algo)

        # Build concepts
        if algo == 'in-close':
            
            global r_new
            global right_degrees
            r_new = 0
            right_degrees = context.I.T.dot(np.ones(context.I.shape[0]))
            extents, intents = init_in_close(context)
            concepts = algo_func(context, extents, intents, r=0, y=0, **kwargs)
        else:
            concepts = algo_func(context)
        
        # Build ConceptLattice
        cl = ConceptLattice(concepts=concepts)
        cl.context = context

        return cl

    def pairwise_concept_distance(self, metric: str = 'Jaccard') -> np.ndarray:
        """Return matrix of pairwise concept distances. The distance between two concepts :math:`X_1` and :math:`X_2` 
        is computed using the mean of the `metric` function between respective extents and intents of the concepts, i.e. 
        :math:`1 - 0.5 * \psi(A_1, A_2) + \psi(B_1, B_2)`, with :math:`\psi` the `metric` function, :math:`A_i` and 
        :math:`B_i` the extent and intent of concept :math:`i`.

        Parameters
        ----------
        metric : str, optional
            Metric used to compute distances between concepts, by default 'Jaccard'

        Returns
        -------
        np.ndarray
            Square matrix of size :math:`n` with :math:`n` the number of concepts in the lattice.
        """        
        n = len(self.concepts)
        i, j = np.triu_indices(n, k=1)
        concepts_arr = np.array(self.concepts, dtype='object')
        print(f'Number of (unique) pairs: {len(i)*2}/2={len(i)}')
        
        metric_func = get_metric(metric)
        distances = []
        
        for k in range(len(i)):
            distances.append(1 - 0.5 * (
                                metric_func(concepts_arr[i][:, 0][k], concepts_arr[j][:, 0][k]) + 
                                metric_func(concepts_arr[i][:, 1][k], concepts_arr[j][:, 1][k])
                                )
                            )
        
        result = np.zeros((n, n))
        result[i, j] = distances
        result = result + result.T
        
        return result

    def top_k(self, k: int = 5, metric: str = 'Jaccard', msg: bool = False) -> list:
        """Return k concepts maximizing `metric`.with highest distance between them.

        Parameters
        ----------
        k : int, optional
            Number of concepts to return, by default 5
        metric : str
            Metric to maximize. Either
                * ``Jaccard``: the Jaccard concept distance is maximized
                * ``MILP``: Multi objective Linear Optimization is performed to find subset of concepts maximizing both
                    the size of their extent and intent.
        msg : bool, optional
            If `True` and `metric` is ``MILP``, print solver log in standard output, by default False

        Returns
        -------
        list
            List of concepts
        """        
        if metric == 'Jaccard':
            distance_mat = self.pairwise_concept_distance()
            idxs = np.argsort(-distance_mat.dot(np.ones(len(self.concepts))))
            concepts_arr = np.array(self.concepts, dtype='object')
            return [(x[0], x[1]) for x in concepts_arr[idxs][:k]]
        
        elif metric == 'MILP':
            solver = Solver(self)
            concepts, _ = solver.solve(k=k, metric='size', msg=msg)
            return [(x[0], x[1]) for x in concepts]

        elif metric == 'tf-idf':
            solver = Solver(self)
            concepts, _ = solver.solve(k=k, metric='tf-idf', msg=msg)
            return [(x[0], x[1]) for x in concepts]
            
    def filter(self, ext_query: dict, int_query: list) -> list:
        """Filter concepts w.r.t user queries.

        Parameters
        ----------
        ext_query : dict
            Dictionary containing filters on extent in keys: 
                * ``degree_left``: filter concepts containing left nodes in extent with degree greater or equal than ``degree_left``
                * ``degree_right``: filter concepts containing right nodes in extent with degree greater or equal than ``degree_right``
                * ``nodes_right``: filter concepts with extent containing set of nodes in ``nodes_right``.
        int_query : list
            List of attributes must contained by filtered concepts.

        Returns
        -------
        list
            List of filtered concepts.
        """        
        concepts = []

        for c in self.concepts:
            g_concept = self.context.graph.adjacency_csr.T[[self.context.G2idx.get(i) for i in c[0]]].T
            idx = np.flatnonzero(get_degrees(g_concept))
            g_concept = g_concept[idx, :]
            
            if np.all(get_degrees(g_concept) >= ext_query.get('degree_left')) \
                and np.all(get_degrees(g_concept, transpose=True) >= ext_query.get('degree_right')) \
                and len(set(ext_query.get('node_right')).intersection(set(c[0]))) >= len(ext_query.get('node_right')) \
                and len(set(int_query).intersection(set(c[1]))) >= len(int_query):
                concepts.append((c[0], c[1]))

        return concepts

def close_by_one(context: FormalContext) -> List[FormalConcept]:
    """
    Parameters
    ----------
    context : FormalContext
        Formal context used to find concepts

    Returns
    -------
    List
        List of formal concepts

    References
    ----------
    O. Kuznetsov (1999). 
    Learning of Simple Conceptual Graphs from Positive and Negative Examples. 
    In J. M. \.Zytkow and J. Rauch, editors, Principles of Data Mining and Knowledge Discovery.
    """
    # Find all concepts using Close by One (CbO) algorithm

    # TODO: add first and last concepts to lattice, i.e empty intent and empty extent concept
    L = []
    
    for g in context.G:
        process([g], context.intention([g]), L, context, g)
    hashes = []
    L_dedup = []
    for c in L:
        h = set(c[0]).union(set(c[1]))
        if h not in hashes:
            hashes.append(h)
            L_dedup.append(c)
    
    return L_dedup

def process(A: list, attrs: list, L: list, context: FormalContext, current_obj: int):
    """Process Close by One algorithm.

    Parameters
    ----------
    A : list
        List of objects
    attrs : list
        List of attributes
    L : list
        List of valid concepts
    context : FormalContext
        Formal Context on which CbO algorithm is applied
    current_obj : int
        Current selected object
    """    

    # (C, D) forms a formal concept
    D = context.intention(A)
    C = context.extension(D, return_names=False)

    A_i = [context.G2idx.get(a) for a in A] # object indexes
    
    # If a concept has already been elaborated, the extent C contains at least one object lexicographically smaller 
    # than the current object -> we verify that (C - A) < current_object is False
    diff = set(C).difference(set(A_i))
    is_lex_smaller = any([h < np.max(context.G2idx.get(current_obj)) for h in diff])
    
    if not is_lex_smaller:

        # Add concept to result
        L.append((list(context.G[C]), D))

        # Candidates: complementary objects of current object
        comp_g = set(np.arange(0, len(context.G))).difference(set(A_i))
        candidates = [h for h in comp_g ]
        
        for f in candidates:
            Z = set(C).union(set([f])) # Add candidate object to extension
            Z_labels = context.G[list(Z)]
            f_intention = context.intention(context.G[[f]])
            Y = set(D).intersection(f_intention) # Compute new intention
            if len(Y) != 0:
                X = context.extension(Y)
                process(Z_labels, f_intention, L, context, current_obj)


def in_close(context: FormalContext, extents: list, intents: list, r: int = 0, y: int = 0, minimum_support: int = 0,
            maximum_support: int = np.inf, max_right_degree: int = 10) -> list:
    """In Close algorithm. 

    Parameters
    ----------
    context : FormalContext
        Formal Context from which concepts are retrieved.
    extents : list
        List of extents filled during the process, i.e objects of concepts.
    intents : list
        List of intents filled during the process, i.e attributes of concepts.
    r : int, optional
        Index of concept being closed, by default 0
    y : int, optional
        Starting attribute index, by default 0
    minimum_support : int, optional
        If over 0, `minimum support` defines the minimum size of interest of extents, by default 0

    Returns
    -------
    List
        List of formal concepts

    References
    ----------
    S. Andrews (2009). 
    In-Close, a fast algorithm for computing formal concepts. 
    In International Conference on Conceptual Structures (ICCS), Moscow.
    """        
    global r_new
    r_new = r_new + 1

    for j in context.M[y:]:
        try:
            extents[r_new] = []
        except IndexError:
            extents.append([])

        #if right_degrees[context.M2idx.get(j)] < max_right_degree:
        if right_degrees[context.M2idx.get(j)] < (0.1 * context.I.shape[1]):

            # Form a new extent by adding extension of attribute j to current concept extent
            extents[r_new] = list(sorted(set(extents[r]).intersection(set(context.extension([j])))))

            # If the extent is empty, skip recursion and move to next attribute. 
            # If the extent is unchanged, add attribute j to current concept intent, skip recursion and move to next attribute.
            # Otherwise, extent must be a smaller (lexicographically) intersection. If the extent already exists, skip recursion and move on to next attribute.
            if (len(extents[r_new]) > minimum_support) and (len(extents[r_new]) < maximum_support):
                if len(extents[r_new]) == len(extents[r]):
                    intents[r] = list(sorted(set(intents[r]).union(set([j]))))
                else:
                    # Test if extent has already be generated using is_cannonical()
                    if is_cannonical(context, extents, intents, r, context.M2idx.get(j) - 1):
                        try:
                            intents[r_new] = []
                        except IndexError:
                            intents.append([])

                        # Extent is cannonical, i.e new concept 
                        # -> Initialize intent and recursive call to begin closure
                        intents[r_new] = list(sorted(set(intents[r]).union(set([j]))))
                        in_close(context, extents, intents, r=r_new, y=context.M2idx.get(j) + 1, 
                                minimum_support=minimum_support)

    return [*zip(extents, intents)]


def is_cannonical(context, extents, intents, r, y):
    global r_new

    for k in range(len(intents[r])-1, -1, -1):
        for j in range(y, context.M2idx.get(intents[r][k]), -1):            
            for h in range(len(extents[r_new])):
                if context.I[context.G2idx.get(extents[r_new][h]), j] == 0:
                    h -= 1 # Necessary for next test in case last interaction of h for-loop returns False
                    break
            if h == len(extents[r_new]) - 1:
                return False
        y = context.M2idx.get(intents[r][k]) - 1

    for j in reversed(range(y, -1, -1)):
        for h in range(len(extents[r_new])):
            if context.I[context.G2idx.get(extents[r_new][h]), j] == 0:
                h -= 1 # Necessary for next test in case last interaction of h for-loop returns False
                break
        if h == len(extents[r_new]) - 1:
            return False
    
    return True
    
def init_in_close(context: FormalContext):
    """In Close algorithm initilization of extent and intent lists.

    Parameters
    ----------
    context : FormalContext
        Formal Context used to retrieve concepts.

    Returns
    -------
    Extent and Intent lists intialized.
    """    
    extents, intents = [], []
    extents_init = list(dict(sorted(context.G2idx.items(), key=lambda x: x[1])).keys())
    intents_init = []
    extents.append(extents_init) # Initalize extents with all objects from context
    intents.append(intents_init) # Initialize intents with empty set attributes
    
    return extents, intents


def get_algo(algo: str) -> object:
    """Return algorithm function according to algorithm name.

    Parameters
    ----------
    algo : str
        Algorithm name

    Returns
    -------
    object
        Algorithm function

    Raises
    ------
    ValueError
        If algorithm name is not known.
    """    
    if algo == 'CbO':
        return close_by_one
    elif algo == 'in-close':
        return in_close
    else:
        raise ValueError(f"Algorithm '{algo}' is not known. Possible values are: 'CbO' (Close by One) or 'in-close' (In-Close).")
