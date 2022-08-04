import numpy as np
import itertools
import random
from tqdm import tqdm

from src.graph import BipartiteGraph


class AdamicAdar:
    def __init__(self):
        self.scores = {}

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
        weight = 0

        for c in commons:
            degree = len(g.get_neighbors(c))
            if degree > 1:
                weight += 1 / (np.log(degree))
            else:
                weight += 0
        
        return weight

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
            hops2 = g.get_neighbors_2hops(node, transpose)
            for i, opp_nodes in tqdm(enumerate(neighbs_opps)):
                score = self._compute_index(g, hops2, opp_nodes)
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