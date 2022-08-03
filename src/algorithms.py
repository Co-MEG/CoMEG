import numpy as np
from typing import Tuple

from src.graph import BipartiteGraph


class AdamicAdar:
    def __init__(self):
        self.scores = {}

    def _compute_index(self, g: BipartiteGraph, nodes_u: str, nodes_v: str):
        """Compute Adamic Adar index between two nodes.

        Parameters
        ----------
        g : BipartiteGraph
            Bipartite Graph
        nodes_u : str
            Target node
        nodes_v : str
            Target node
        """        
        commons = set(nodes_u).intersection(set(nodes_v))
        weight = 0
        for c in commons:
            degree = len(g.get_neighbors(c))
            if degree > 1:
                weight += 1 / (np.log(degree))
            else:
                weight += 0
            self.scores.append(weight)

    def predict(self, g: BipartiteGraph, use_left: bool = True) -> list:
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
        if use_left:
            left_nodes = g.V['left']
            right_nodes = g.V['right']
        else:
            left_nodes = g.V['right']
            right_nodes = g.V['left']
        
        for left_n in left_nodes:
            hops2 = [] # TODO: implement hops2 function
            for right_n in right_nodes:
                neighbs_right_n = g.neighbors(right_n)
                # Compute index only for edges that do not exist
                if (left_n, right_n) not in g.E:
                    score = self._compute_index(g, hops2, neighbs_right_n)
                    self.scores.update({(left_n, right_n): score})
                    
        return self.scores