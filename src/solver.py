import numpy as np
from pulp import *

class Solver():
    def __init__(self, concepts: list):
        self.concepts = np.array(concepts, dtype='object')
        self.concepts_idx = np.arange(0, len(self.concepts))
        self.x = pulp.LpVariable.dicts("x", 
                                indices = self.concepts_idx, 
                                lowBound=0, upBound=1, 
                                cat='Integer', indexStart=[])
    
    def model(self, k: int = 5):
        """Build optmization model with objective and constraints.

        Parameters
        ----------
        k : int, optional
            Number of concepts to select, by default 5

        Returns
        -------
        ``pulp`` problem object.
        """        
        # Variables are lengths of extent and intent of concepts
        ext_len, int_len = [], []
        for c in self.concepts:
            ext_len.append(len(c[0]))
            int_len.append(len(c[1]))

        # Find subset of 5 concepts that maximize both lengths of extents and intents
        prob = pulp.LpProblem("Best_concepts", LpMaximize)
        prob += pulp.lpSum([self.x[i] * (ext_len[i] + int_len[i]) for i in self.concepts_idx])
        prob += pulp.lpSum([self.x[i] for i in self.concepts_idx]) == k
        
        return prob

    def solve(self, k: int = 5, solver: object = PULP_CBC_CMD, msg: bool = False) -> list:
        """Solve optimization problem (default solver is CBC MILP).

        Parameters
        ----------
        k : int, optional
            Number of concepts to select, by default 5
        msg : bool, optional
            If `True`, print solver log in standard output, by default False

        Returns
        -------
        list
            List of concepts maximizing optimization problem.
        """        
        prob = self.model(k=k)
        prob.solve(solver(msg=msg))

        concept_idxs = np.array([int(v.name.split('_')[1]) for v in prob.variables() if v.varValue == 1])
        
        return list(self.concepts[concept_idxs])
