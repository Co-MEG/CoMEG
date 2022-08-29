import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
import numpy as np
from pulp import *
import pandas as pd
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
            int_len.append(len(c[1]) / len(self.lattice.context.M))

            # Size of extent
            #ext_len.append(len(c[0]))

            # Size of graph induced by concept
            attr_idxs = [self.lattice.context.G2idx.get(i) for i in c[0]]
            g_concept = self.lattice.context.graph.adjacency_csr.T[attr_idxs]
            n_right, n_left = g_concept.shape
            m = g_concept.nnz
            ext_len.append((n_right + n_left + m) / self.lattice.context.graph.size())

            print(round((n_right + n_left + m) / self.lattice.context.graph.size(), 3),
                round(len(c[1]) / len(self.lattice.context.M), 3),
                c)

        return ext_len, int_len
    
    def multi_obj_model(self, k: int = 5):
        # Variables are lengths of extent and intent of concepts
        ext_len, int_len = self.init_model_variables()
        # Find subset of 5 concepts that maximize both lengths of extents and intents
        
        step_size = 0.1
        solutionTable = pd.DataFrame(columns=["alpha", "obj_value"])
        probs = []
        objs = []
        min_obj = np.inf
        for alpha in np.arange(0, 1 + step_size, step_size):
            prob = pulp.LpProblem("Best_concepts Multi objectives", LpMaximize)
            #prob += pulp.lpSum([self.x[c] * (alpha * ext_len[c] + (1 - alpha) * int_len[c]) for c in self.concepts_idx])
            prob += alpha * pulp.lpSum([self.x[c] * ext_len[c] for c in self.concepts_idx]) \
                + (1 - alpha) * pulp.lpSum([self.x[c] * int_len[c] * 3 for c in self.concepts_idx])
            prob += pulp.lpSum([self.x[c] for c in self.concepts_idx]) == k
            
            solution = prob.solve(PULP_CBC_CMD(msg=False))
            solutionTable.loc[int(alpha*1/step_size)] = [alpha, pulp.value(prob.objective)]
            if pulp.value(prob.objective) <= min_obj:
                min_obj = pulp.value(prob.objective)
                min_prob = prob
                min_alpha = alpha
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        print(solutionTable)
        plt.plot(solutionTable['alpha'], solutionTable['obj_value'], color='g')
        plt.xlabel('alpha')
        plt.ylabel('Objective value')
        #plt.show()
        PATH_RES = os.path.join(os.getcwd(), 'data', 'goodreads_poetry', 'result')
        res = os.path.join(PATH_RES, 'img', f'LO_pareto_bc1d727746e210f315138932e0aacb11_13637887.eps')
        plt.tight_layout()
        plt.savefig(res)

        print(f'Alpha selected: {min_alpha}')

        return min_prob

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
        
        # Multiple objective optimization
        prob = self.multi_obj_model(k=k)

        # Simple objective optimization
        """prob = self.model(k=k)
        prob.solve(solver(msg=msg))"""
        
        concept_idxs = np.array([int(v.name.split('_')[1]) for v in prob.variables() if v.varValue == 1])
        scores = np.array([v.dj for v in prob.variables()])

        return list(self.concepts[concept_idxs]), scores
