import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm


def d2v_embedding(model, doc):
    """Use pre-trained model to embed document."""
    return model.infer_vector(doc)

def pairwise_wd_distance(matrix, n, model, names):
    """Doc2Vec embedding + pairwise Wasserstein distances between elements in matrix."""

    wd_matrix = np.zeros((n, n))
    
    for i in tqdm(range(n)):
        w1 = d2v_embedding(model, names[np.flatnonzero(matrix[i, :])])
        for j in range(n):
            w2 = d2v_embedding(model, names[np.flatnonzero(matrix[j, :])])
            wd_matrix[i, j] = wasserstein_distance(w1, w2)
            
    return wd_matrix