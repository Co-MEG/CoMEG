from src.context import FormalContext
from src.concept import FormalConcept

from typing import List
import numpy as np


class ConceptLattice:
    def __init__(self, concepts: List[FormalConcept]):
        # Build a Concept Lattice given formal concepts
        self.concepts = concepts

    @classmethod
    def from_context(cls, context: FormalContext, algo: str = 'CbO') -> 'ConceptLattice':
        """Return a ``ConceptLattice`` according to ``context``, created using algorithm ``algo``.

        Parameters
        ----------
        context : FormalContext
            Formal context on which concept lattice is created.
        algo : str, optional
            Algorithm name, by default 'CbO' (Close by One)

        Returns
        -------
        ConceptLattice
            Concept Lattice
        """        
        algo_func = get_algo(algo)

        # Build concepts
        concepts = algo_func(context)
        tot = []
        for c in concepts:
            if c not in tot:
                tot.append(c)
        for c in tot:
            print(c)
        print(f"# of concepts: {len(tot)}")

        # Build ConceptLattice
        cl = ConceptLattice(concepts=tot)

        return cl


def close_by_one(context: FormalContext) -> List[FormalConcept]:
    """
    References
    ----------
    O. Kuznetsov (1999). 
    Learning of Simple Conceptual Graphs from Positive and Negative Examples. 
    In J. M. \.Zytkow and J. Rauch, editors, Principles of Data Mining and Knowledge Discovery.
    """
    # Find all concepts using Close by One (CbO) algorithm
    L = []
    for i, g in enumerate(context.G):
        process([g], context.intention([g]), L, context, g)
    return L

def process(A, attrs, L, context, g):
    print(f"**Process({A}, {attrs})")

    #D = attrs
    D = context.intention(A)
    C = context.extension(D, return_names=False)
    print(f"  C (extent): {context.G[C]}")
    print(f"  D (intent): {D}")

    A_i = [context.G2idx.get(a) for a in A]
    
    # If a concept has already been elaborated, the extent C contains at least one object lexicographically smaller 
    # than the current object -> we verify that C < A_i is False
    #print(f'  Test if {context.G[C]} is lex less than {context.G[A_i]} : {minus_lex(set(C), set(A_i))}')
    #print(f'  Test if {set} is lex less than {context.extension(attrs, return_names=False)}')
    
    # new
    #diff = set(A_i).difference(set(C))
    #is_lex_smaller = any([h < np.max(context.extension(attrs, return_names=False)) for h in diff])
    #if not is_lex_smaller:

    # newnew
    diff = set(C).difference(set(A_i))
    is_lex_smaller = any([h < np.max(context.G2idx.get(g)) for h in diff])
    print(f'  Test if {diff} (i.e {C}-{A_i}) is lex less than {np.max(context.G2idx.get(g))}')
    if not is_lex_smaller:
    #if not minus_lex(set(C), set(A_i)):
        print(f'  New extent: {context.G[C]}')

        L.append((list(context.G[C]), D))
        print(f'  Added concept: {(context.G[C], D)}')

        # Complementary objects of current object, which are lexicographically above the current object
        comp_g = set(np.arange(0, len(context.G))).difference(set(A_i))
        #candidates = [h for h in comp_g if minus_lex(set(A_i), set([h]))]
        
        # new
        #candidates = [h for h in comp_g if h > np.max(context.extension(attrs, return_names=False))]
        candidates = [h for h in comp_g ]
        print(f'  Try to add some more candidates in the extent...')
        for f in candidates:
            print(f'    Candidate: {context.G[f]}')
            Z = set(C).union(set([f])) # Add candidate object to extension
            Z_labels = context.G[list(Z)]
            print(f'    Z: {Z_labels}')
            f_intention = context.intention(context.G[[f]])
            Y = set(D).intersection(f_intention) # Compute new intention
            print(f'    Y: {Y}')
            if len(Y) != 0:
                X = context.extension(Y)
                print(f'    X: {X}')
                process(Z_labels, f_intention, L, context, g)

    #print(f'  L : {(L)}')
    print()

def minus_lex(A: set, B: set) -> bool:
    diff = sorted(A.difference(B))
    return any([x < sorted(B)[-1] for x in diff])
    #diff = sorted(B.difference(A))
    #return len(diff) > 0 and \
    #    any([A.intersection(set(np.arange(1, i))) == B.intersection(set(np.arange(1, i))) for i in diff])



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
    else:
        raise ValueError(f"Algorithm '{algo}' is not known. Possible values are: 'CbO' (Close by One).")
