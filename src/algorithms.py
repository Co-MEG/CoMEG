import numpy as np
import random
from tqdm import tqdm
import time
import os
import json
#from multiprocessing import Pool
from alive_progress import alive_bar

from src.graph import BipartiteGraph


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