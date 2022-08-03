import numpy as np
from typing import Tuple

from src.graph import BipartiteGraph


class AdamicAdar:
    def __init__(self):
        self.scores = {}

    def _compute_index(self, g, nodes_u, nodes_v):
        commons = set(nodes_u).intersection(set(nodes_v))
        weight = 0
        for c in commons:
            degree = len(g.neighbors(c))
            if degree > 1:
                weight += 1 / (np.log(degree))
            else:
                weight += 0
            self.scores.append(weight)

    def predict(self, g: BipartiteGraph, use_left: bool = True) -> Tuple:

        if use_left:
            left_nodes = g.V['left']
            right_nodes = g.V['right']
        else:
            left_nodes = g.V['right']
            right_nodes = g.V['left']
        
        for left_n in left_nodes:
            hops2 = []
            for right_n in right_nodes:
                neighbs_right_n = g.neighbors(right_n)
                # Compute index only for edges that do not exist
                if (left_n, right_n) not in g.E:
                    score = self._compute_index(g, hops2, neighbs_right_n)
                    self.scores.update({(left_n, right_n): score})
        return self.scores