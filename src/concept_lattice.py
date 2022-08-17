from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

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
    def from_context(cls, context: FormalContext, algo: str = 'CbO') -> ConceptLattice:
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
