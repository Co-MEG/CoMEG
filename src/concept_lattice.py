from src.context import FormalContext
from src.concept import FormalConcept

from typing import List


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

        References
        ----------
        O. Kuznetsov (1999). 
        Learning of Simple Conceptual Graphs from Positive and Negative Examples. 
        In J. M. \.Zytkow and J. Rauch, editors, Principles of Data Mining and Knowledge Discovery.
        """        
        algo_func = get_algo(algo)

        if algo == 'CbO':
            # Build concepts
            concepts = algo_func(context)
            print(concepts)

        # Build ConceptLattice
        cl = ConceptLattice(concepts)

        return cl


def close_by_one(context: FormalContext) -> List[FormalConcept]:
    # Find all concepts using Close by One (CbO) algorithm
    pass

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

