from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

from src.concept import FormalConcept
if TYPE_CHECKING:
    from src.context import FormalContext


class ConceptLattice:
    def __init__(self, concepts: List[FormalConcept]):
        # Build a Concept Lattice given formal concepts
        self.concepts = concepts

    def __len__(self) -> int:
        """Return number of concepts in Lattice.

        Returns
        -------
        int
            Number of concepts
        """        
        return len(self.concepts)

    @classmethod
    def from_context(cls, context: FormalContext, algo: str = 'in-close') -> ConceptLattice:
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
            r_new = 0
            extents, intents = init_in_close(context)
            concepts = algo_func(context, extents, intents, r=0, y=0)
        else:
            concepts = algo_func(context)
        
        # Build ConceptLattice
        cl = ConceptLattice(concepts=concepts)

        return cl


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


def in_close(context: FormalContext, extents: list, intents: list, r: int = 0, y: int = 0, minimum_support: int = 0) \
    -> list:
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
    print(f'**Idx of current concept: {r} {extents[r], intents[r]} - inclose(r={r},y={y})**')
    print('---------------------------------')
    print(f'  idx candidate new concept r_new: {r_new} - columns to try: {context.M[y:]}')
    for j in context.M[y:]:
        print(f'     Attribute {j} (idx={context.M2idx.get(j)}) - Current concept: {r} ')
        try:
            extents[r_new] = []
        except IndexError:
            extents.append([])

        # Form a new extent by adding extension of attribute j to current concept extent
        print(f'        Extent r new: {set(extents[r_new])} - extents r: {extents[r]}')
        extents[r_new] = list(sorted(set(extents[r]).intersection(set(context.extension([j])))))
        print(f'        Extent r new after : {sorted(set(extents[r_new]))}')

        # If the extent is empty, skip recursion and move to next attribute. 
        # If the extent is unchanged, add attribute j to current concept intent, skip recursion and move to next attribute.
        # Otherwise, extent must be a smaller (lexicographically) intersection. If the extent already exists, skip recursion and move on to next attribute.
        if len(extents[r_new]) > minimum_support:
            
            if len(extents[r_new]) == len(extents[r]):
                print(f'        Extent is unchanged -> modify only intent of concept r')
                intents[r] = list(sorted(set(intents[r]).union(set([j]))))
                print(f'        Intent[{r}] after : {intents[r]}')
            else:
                print(f'        RECURSION -> is_canno({r},{context.M2idx.get(j) - 1})?')
                # Test if extent has already be generated using is_cannonical()
                if is_cannonical(context, extents, intents, r, context.M2idx.get(j) - 1):
                    print(f'        ==>Is cannonical')
                    try:
                        intents[r_new] = []
                    except IndexError:
                        intents.append([])
                    
                    # Extent is cannonical, i.e new concept 
                    # -> Initialize intent and recursive call to begin closure
                    intents[r_new] = list(sorted(set(intents[r]).union(set([j]))))
                    print(f'        New intents[{r_new}]: {intents[r_new]} - CALL INCLOSE')
                    in_close(context, extents, intents, r=r_new, y=context.M2idx.get(j) + 1, 
                            minimum_support=minimum_support)
    print(f'ENDFOR -> return to previous call of in-close')
    return [*zip(extents, intents)]


def is_cannonical(context, extents, intents, r, y):
    global r_new

    #for k in reversed(range(len(intents[r]))):
    print(f'        |B[r={r}]-1|={len(intents[r])-1}')
    
    # \o/ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for k in range(len(intents[r])-1, -1, -1):
        
        print(f'        k: {k} |B[r]|={len(intents[r])}')
        #for j in intents[r][y:k+1:-1]:
        for j in range(y, context.M2idx.get(intents[r][k]), -1):
            print(f'          j:{j} -> downto B[r={r}][k={k}]={intents[r][k]}, idx={context.M2idx.get(intents[r][k])}')
            
            for h in range(len(extents[r_new])):
                print(f'            h={h} - context.I[A[r_new={r_new}][h={h}], j={j}] = I[{extents[r_new][h]}][{context.M[j]}] = {context.I[context.G2idx.get(extents[r_new][h]), j]}')
                #if not context.I[context.G2idx.get(extents[r_new][h]), context.M2idx.get(j)]:
                if context.I[context.G2idx.get(extents[r_new][h]), j] == 0:
                    h -= 1 # Necessary for next test in case last interaction of h for-loop returns False
                    break
            #if h == len(extents[r_new]) - 1:
            if h == len(extents[r_new]) - 1:
                print(f'        Not canonical! -> stop recursion')
                return False
        y = context.M2idx.get(intents[r][k]) - 1

    for j in reversed(range(y, -1, -1)):
        print(f'        j: {j} - |extents[r_new={r_new}]|={len(extents[r_new])}')
        for h in range(len(extents[r_new])):
            print(f'          h={h} - context.I[A[r_new={r_new}][h={h}], j={j}] = I[{extents[r_new][h]}][{context.M[j]}] = {context.I[context.G2idx.get(extents[r_new][h]), j]}')
            #if not context.I[context.G2idx.get(extents[r_new][h]), j]:
            if context.I[context.G2idx.get(extents[r_new][h]), j] == 0:
                h -= 1 # Necessary for next test in case last interaction of h for-loop returns False
                break
        #if h == len(extents[r_new]) - 1:
        print(f'        h={h} = |A[r_new]|-1={len(extents[r_new])-1}? {h == len(extents[r_new])-1}')
        if h == len(extents[r_new]) - 1:
            print(f'        Not canonical bis! -> stop recursion')
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
