import numpy as np

def concept_attributes(biadjacency, labels):
    """Build concept x attributes matrix. Column values are count of occurrences of attributes for each concept/community.
    
    Parameters
    ----------
    biadjacency: sparse.csr_matrix
        Biadjacency matrix of the graph
    labels: np.ndarray
        Belonging community for each node in the graph, e.g Louvain labels or KMeans labels
        
    Outputs
    -------
        Matrix with concepts/communities in rows and count of attributes in columns. """

    nb_cc = len(np.unique(labels))
    matrix = np.zeros((nb_cc, biadjacency.shape[1]))
    for c in range(nb_cc):
        mask_cc = labels == c
        indices_attr = biadjacency[mask_cc].indices
        for ind in indices_attr:
            matrix[c, ind] += 1

    return matrix

def build_concept_attributes(result, biadjacency, labels_cc_summarized, labels_louvain, kmeans_gnn_labels, 
                                kmeans_spectral_labels, kmeans_doc2vec_labels):

    # patterns from Unexpectedness algorithm 
    patterns_attributes = np.zeros((len(result), biadjacency.shape[1]))
    for i, c in enumerate(result[1:]):
        patterns_attributes[i, c[1]] = 1
    
    # Concept x attributes matrices for all methods
    concept_summarized_attributes = concept_attributes(biadjacency, labels_cc_summarized)
    concept_louvain_attributes = concept_attributes(biadjacency, labels_louvain)
    concept_gnn_kmeans_attributes = concept_attributes(biadjacency, kmeans_gnn_labels)
    concept_spectral_kmeans_attributes = concept_attributes(biadjacency, kmeans_spectral_labels)
    concept_doc2vec_kmeans_attributes = concept_attributes(biadjacency, kmeans_doc2vec_labels)

    return patterns_attributes, concept_summarized_attributes, concept_louvain_attributes, concept_gnn_kmeans_attributes, concept_spectral_kmeans_attributes, concept_doc2vec_kmeans_attributes
