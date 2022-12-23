from typing import Union
import numpy as np

def cosine_similarity(c1: Union[list, set], c2: Union[list, set]) -> float:
    return c1.dot(c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))

def jaccard_score(c1: Union[list, set], c2: Union[list, set]) -> int:
    """Jaccard similarity coefficient score, i.e. the size of the intersection divided by the size of the union of two
    label sets.

    :math:`J(A,B)=\dfrac{A \cap B}{A \cup B}`

    Parameters
    ----------
    c1 : Union[list, set]
        First list or set of values.
    c2 : Union[list, set]
        Second list or set of values.

    Returns
    -------
    int
        _description_
    """    
    if not isinstance(c1, set):
        c1 = set(c1)
    if not isinstance(c2, set):
        c2 = set(c2)
    return len(c1.intersection(c2)) / len(c1.union(c2))

def get_metric(metric: str) -> object:
    """Return metric function according to metric name.

    Parameters
    ----------
    metric : str
        Metric name

    Returns
    -------
    object
        Metric function

    Raises
    ------
    ValueError
        If metric name is not known.
    """    
    if metric == 'Jaccard':
        return jaccard_score
    else:
        raise ValueError(f"Metric '{metric}' is not known. Possible values are: 'Jaccard' (Jaccard index).")
