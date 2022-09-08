from alive_progress import alive_bar
import json
import numpy as np
import os
import random
from scipy import sparse
from tqdm import tqdm
from typing import Tuple

from sknetwork.embedding import Spectral

from src.graph import BipartiteGraph

class TfIdf():
    def __init__(self):
        self.n = None
        self.m = None

    def _idf(self, biadjacency: sparse.csr_matrix) -> np.ndarray:
        """Compute inverse document frequency for each word columns of `biadjacency`.

        Parameters
        ----------
        biadjacency : sparse.csr_matrix
            Rows are documents, words are columns

        Returns
        -------
        Array of inverse document frequency value for each word in biadjacency.
        """        
        count = biadjacency.T.astype(bool).dot(np.ones(self.n))
        
        return np.log((self.n) / (count+1))

    def _tf(self, biadjacency: sparse.csr_matrix) -> np.ndarray:
        """Compute term frequency for each word in each row of `biadjacency`.

        Parameters
        ----------
        biadjacency : sparse.csr_matrix
            Rows are documents, words are columns

        Returns
        -------
        Array of term frequencies with same size as `biadjacency`
        """        
        diag = sparse.diags(biadjacency.dot(np.ones(self.m)))
        diag.data = 1 / diag.data

        return diag.dot(biadjacency)

    def fit_transform(self, biadjacency: sparse.csr_matrix) -> np.ndarray:
        """Compute tf-idf.

        Parameters
        ----------
        biadjacency : sparse.csr_matrix
            Rows are documents, words are columns

        Returns
        -------
        Array of tf-idf values for each word in each document.
        """        
        self.n, self.m = biadjacency.shape
        diag_idf = sparse.diags(self._idf(biadjacency))

        return self._tf(biadjacency).dot(diag_idf)


class SpectralEmb:
    def __init__(self, k: int = 40, normalized: bool = True):
        """Spectral embedding of graphs.

        Parameters
        ----------
        k : int, optional
            Embedding dimension, by default 40
        normalized : bool, optional
            If `True`, normalize embeddings so that each vector has norm 1 in embedding space, by default True
        """    
        self.k = k    
        self.normalized = normalized
        self.train_g = None
        self.test_g = None
        self.embedding_row = None
        self.embedding_col = None

    def fit_transform(self, train_g: BipartiteGraph) -> np.ndarray:
        """Fit algorithm to data an return embeddings and return row embeddings.

        Parameters
        ----------
        train_g : BipartiteGraph
            Bipartite graph

        Returns
        -------
        Array of row embeddings
        """        
        self.train_g = train_g
        
        spectral = Spectral(self.k, normalized=self.normalized)
        self.embedding_row = spectral.fit_transform(train_g.adjacency_csr)
        self.embedding_col = spectral.embedding_col_

        return self.embedding_row

    def predict_edges(self, test_g: BipartiteGraph, save_result: bool = True, path: str = None) -> Tuple:
        """Predict existence of an edge between two nodes by computing the cosine similarity between their representation
        in the embedding space.

        Parameters
        ----------
        test_g : BipartiteGraph
            Bipartite graph (usually test graph)
        save_result : bool, optional
            If `True`, save result to ``path``, by default True
        path : str, optional
            Path used to save result, by default None

        Returns
        -------
        Tuple
            Tuple of arrays containing ground truth values and predicted values for each edge in `test_g`
        """                            
        self.test_g = test_g

        # Cosine similarity between node representation (need normalized=True in fit_transform, otherwise dot-product)
        g_coo = self.test_g.adjacency_csr.tocoo()
        
        # Positive graph
        scores_pos = (self.embedding_row[g_coo.row] * self.embedding_col[g_coo.col]).sum(axis=1)

        # Negative graph -> sampling random edges
        rand_n_right = np.random.randint(self.embedding_col.shape[0], size=len(scores_pos))
        scores_neg = (self.embedding_row[g_coo.row] * self.embedding_col[rand_n_right]).sum(axis=1)
        y_true_neg = []
        # Verify if edge exists in original graph (i.e. train+test)
        idxs = []
        for idx, (src, dst) in enumerate(zip(g_coo.row, rand_n_right)):
            edge_exist = max(self.train_g.adjacency_csr[src, dst], self.test_g.adjacency_csr[src, dst])
            if edge_exist == 0:
                y_true_neg.append(edge_exist)
                idxs.append(idx)
        scores_neg = scores_neg[np.array(idxs)]

        y_pred = np.hstack((scores_pos, scores_neg))
        y_true = np.hstack((np.ones(len(scores_pos)), y_true_neg))

        if save_result:
            with open(path, 'w') as f:
                i = 0
                for idx, (u, v, s) in enumerate(zip(self.train_g.names_row[g_coo.row], 
                                                    self.train_g.names_col[g_coo.col], 
                                                    scores_pos)):
                    json.dump({idx: (u, v, s)}, f)
                    f.write(os.linesep)
                    i = idx
                for u, v, s in zip(self.train_g.names_row[g_coo.row], 
                                   self.train_g.names_col[rand_n_right], 
                                   scores_neg):
                    i += 1
                    json.dump({i: (u, v, s)}, f)
                    f.write(os.linesep)
        
        return y_true, y_pred


class AdamicAdar:
    def __init__(self):
        self.scores = {}

    def _log_inv_deg(self, g: BipartiteGraph, node: str) -> float:
        weight = 0
        if node in g.degrees:
            degree = g.degrees.get(node)
        else:
            degree = len(g.get_neighbors(node))
        if degree > 1:
            weight += 1 / (np.log(degree))
        else:
            weight += 0

        return weight

    def _compute_index(self, g: BipartiteGraph, nodes_u: list, nodes_v: list):
        """Compute Adamic Adar index between two nodes.

        Parameters
        ----------
        g : BipartiteGraph
            Bipartite Graph
        nodes_u : list
            List of target nodes
        nodes_v : str
            List of target nodes
        """        
        commons = set(nodes_u).intersection(set(nodes_v))

        # Sequential
        weight = 0

        for c in commons:
            if c in g.degrees:
                degree = g.degrees.get(c)
            else:
                degree = len(g.get_neighbors(c))
            
            if degree > 1:
                weight += 1 / (np.log(degree))
            else:
                weight += 0

        # MultiProcessing
        #start = time.time()
        #with Pool(4) as p:
        #    res = p.starmap(self._log_inv_deg, zip([g] * len(commons), commons))
        #print(f'End multiprocessing: {time.time() - start}')
        
        return weight

    def predict_edges(self, g: BipartiteGraph, edges: list, transpose: bool = False) -> dict:
        """Predict Adamic Adar index for selected edges.

        Parameters
        ----------
        g : BipartiteGraph
            Bipartite graph
        edges : list
            List of edges on which predict index
        transpose : bool, optional
            If `True`, transpose adjacency matrix, by default False

        Returns
        -------
        dict
            Scores for predicted edges. Each entry is a list [u, v, s], with (u, v) the edge and s the predicted score.
        """        
        PATH_RES = os.path.join(os.getcwd(), 'data', 'goodreads_poetry', 'result')
        res = os.path.join(PATH_RES, 'adamic_adar_test_graph.json')
        cpt = 0
        with open(res, "a") as f:
        
            with alive_bar(len(edges)) as bar:
                for e in edges:
                    u, v = e[0], e[1]
                    
                    n_2hops = g.get_neighbors_2hops(u, transpose=transpose)
                    n_opp = g.get_neighbors(v, transpose=~transpose)
                    
                    score = self._compute_index(g, n_2hops, n_opp)
                    self.scores.update({cpt: (u, v, score)})
                    cpt += 1
                    bar()
                    json.dump({cpt: (u, v, score)} , f)
                    f.write(os.linesep)

        return self.scores
    
    def predict_sample(self, g: BipartiteGraph, nodes: list, transpose: bool = False, alpha: float = 0.0001) -> dict:
        """Predict Adamic Adar index for sampled pairs of nodes.

        Parameters
        ----------
        g : BipartiteGraph
            Bipartite Graph
        nodes : list
            List of target nodes
        transpose: bool
            If `True`, column nodes are used for target nodes, by default False
        alpha : float
            Proportion of opposite set nodes to test for index computing, by default 0.0001

        Returns
        -------
        dict
            Scores for sampled edges.
        """        
        e_target_size = int(alpha * g.number_of_nodes())   
        sample_edges = random.sample(g.E, k=int(e_target_size/2))
        
        # Sample opposite nodes
        if transpose:    
            samp_exist_opp_nodes = set([e[0] for e in sample_edges])
            samp_rand_opp_nodes = set(random.sample(g.V['left'], k=int(e_target_size/2)))
        else:
            samp_exist_opp_nodes = set([e[1] for e in sample_edges])
            samp_rand_opp_nodes = set(random.sample(g.V['right'], k=int(e_target_size/2)))
        
        samp_opp_nodes = list(samp_exist_opp_nodes.union(samp_rand_opp_nodes))
        neighbs_opps = [g.get_neighbors(opp_node, ~transpose) for opp_node in tqdm(samp_opp_nodes)]

        cpt = 0
        # Compute index
        for node in tqdm(nodes):
            n_2hops = g.get_neighbors_2hops(node, transpose)
            for i, opp_nodes in tqdm(enumerate(neighbs_opps)):
                score = self._compute_index(g, n_2hops, opp_nodes)
                self.scores.update({cpt: (node, samp_opp_nodes[i], score)})
                cpt += 1
        
        return self.scores

    def predict(self, g: BipartiteGraph, transpose: bool = False) -> list:
        """Predict Adamic Adar index for every pair of nodes that do no form an edge in the input graph.

        Parameters
        ----------
        g : BipartiteGraph
            Bipartite Graph
        use_left : bool, optional
            If `True`, left-sided nodes are used for index computing, by default True

        Returns
        -------
        List
            Scores for pair of nodes that do not form an edge.
        """        
        if not transpose:
            left_nodes = g.V['left']
            right_nodes = g.V['right']
        else:
            left_nodes = g.V['right']
            right_nodes = g.V['left']

        for left_n in tqdm(left_nodes):
            for right_n in tqdm(right_nodes):            
                # Compute index only for edges that do not exist
                #if (left_n, right_n) not in g.E:
                hops2 = g.get_neighbors_2hops(left_n, transpose)
                neighbs_right_n = g.get_neighbors(right_n, ~transpose)
                score = self._compute_index(g, hops2, neighbs_right_n)
                self.scores.update({(left_n, right_n): score})

        return self.scores