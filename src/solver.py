from typing import TYPE_CHECKING
import numpy as np
from pulp import *
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.concept_lattice import ConceptLattice


class Solver():
    def __init__(self, lattice: 'ConceptLattice'):
        self.lattice = lattice
        self.concepts = np.array(lattice.concepts, dtype='object')
        self.concepts_idx = np.arange(0, len(self.concepts))
        self.x = pulp.LpVariable.dicts("x", 
                                indices = self.concepts_idx, 
                                lowBound=0, upBound=1, 
                                cat='Integer', indexStart=[])
    
    def init_model_variables(self):
        ext_len, int_len = [], []
        for c in self.concepts:
            
            # Size of intent
            int_len.append(len(c[1]))

            # Size of extent
            #ext_len.append(len(c[0]))

            # Size of graph induced by concept
            attr_idxs = [self.lattice.context.G2idx.get(i) for i in c[0]]
            g_concept = self.lattice.context.graph.adjacency_csr.T[attr_idxs]
            n_right, n_left = g_concept.shape
            m = g_concept.nnz
            ext_len.append(n_right + n_left + m)

        return ext_len, int_len
    
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
        ext_len, int_len = self.init_model_variables()

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
