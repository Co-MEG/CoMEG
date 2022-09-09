import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Tuple
import numpy as np
from pulp import *
import pandas as pd
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.concept_lattice import ConceptLattice
from src.algorithms import TfIdf

class Solver():
    def __init__(self, lattice: 'ConceptLattice'):
        self.lattice = lattice
        self.concepts = np.array(lattice.concepts, dtype='object')
        self.concepts_idx = np.arange(0, len(self.concepts))
        self.x = pulp.LpVariable.dicts("x", 
                                indices = self.concepts_idx, 
                                lowBound=0, upBound=1, 
                                cat='Integer', indexStart=[])
    
    def init_model_variables(self, metric: str = 'size') -> Tuple:
        """Initialize ``Pulp`` optimization variables.
            
        Parameters
        ----------
        metric : str
            * 'size': Size of extent and intent. Variables are:
                * Coverage of graph induced by the extent wrt the graph in the vicinity of the prediction
                * Ratio between size of the intent and number of unique attributes
            * 'tf-idf': Sum of `tf-idf` scores for each object-attribute pair

        Returns
        -------
        Tuple of lists containing initialized variables.
        """        
        ext_var, int_var = [], []
        if self.lattice.context.use_description:
            n_unique_attr = len(self.lattice.context.M)
        else:
            n_unique_attr = len(np.unique([x.split('_')[:-1] for x in self.lattice.context.M]))

        for c in self.concepts:
            
            if metric == 'tf-idf':
                tfidf_val = 0
                idx_row = [self.lattice.context.G2idx.get(obj) for obj in c[0]]
                idx_col = [self.lattice.context.M2idx.get(attr) for attr in c[1]]
                tfidf_val = self.lattice.context.I[idx_row, :][:, idx_col].sum()
                int_var.append(tfidf_val)
                ext_var.append(np.exp(-len(c[1])/4)) # size of intent is stored in ext_var 

            else:
                int_var.append(len(c[1]) / n_unique_attr)
                # Size of graph induced by concept
                attr_idxs = [self.lattice.context.G2idx.get(i) for i in c[0]]
                g_concept = self.lattice.context.graph.adjacency_csr.T[attr_idxs]
                n_right, n_left = g_concept.shape
                m = g_concept.nnz
                ext_var.append((n_right + n_left + m) / self.lattice.context.graph.size())

        return ext_var, int_var
    
    def multi_obj_model(self, k: int = 5, metric: str = 'size'):
        """Build optmization multi-objective model with constraints, using weighted sum approach.

        Parameters
        ----------
        k : int, optional
            Number of concepts to select, by default 5
        metric : str
            * 'size': Size of extent and intent. Variables are:
                * Coverage of graph induced by the extent wrt the graph in the vicinity of the prediction
                * Ratio between size of the intent and number of unique attributes
            * 'tf-idf': Sum of `tf-idf` scores for each object-attribute pair

        Returns
        -------
        ``pulp`` problem object.
        """        
        # Variables are lengths of extent and intent of concepts
        ext_var, int_var = self.init_model_variables(metric=metric)
        
        # Find subset of 5 concepts that maximize both lengths of extents and intents
        step_size = 0.1
        solutionTable = pd.DataFrame(columns=["alpha", "obj_value"])
        min_obj = np.inf

        for alpha in np.arange(0, 1 + step_size, step_size):
            # Model definition
            prob = pulp.LpProblem("Best_concepts Multi objectives", LpMaximize)
            prob += alpha * pulp.lpSum([self.x[c] * ext_var[c] for c in self.concepts_idx]) \
                + (1 - alpha) * pulp.lpSum([self.x[c] * int_var[c] for c in self.concepts_idx])
            prob += pulp.lpSum([self.x[c] for c in self.concepts_idx]) == k
            if metric == 'size':
                for c in self.concepts_idx:
                    prob += self.x[c] * ext_var[c] <= 0.95
                    prob += self.x[c] * int_var[c] <= 0.95
            # Solving model
            solution = prob.solve(PULP_CBC_CMD(msg=False))
            solutionTable.loc[int(alpha*1/step_size)] = [alpha, pulp.value(prob.objective)]
            
            if pulp.value(prob.objective) <= min_obj:    
                min_obj = pulp.value(prob.objective)
                min_prob = prob

        # Plot Pareto frontier
        #fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        #print(solutionTable)
        #plt.plot(solutionTable['alpha'], solutionTable['obj_value'], color='g')
        #plt.xlabel('alpha')
        #plt.ylabel('Objective value')
        #plt.show()
        # Save result img
        #PATH_RES = os.path.join(os.getcwd(), 'data', 'goodreads_poetry', 'result')
        #res = os.path.join(PATH_RES, 'img', f'LO_pareto_bc1d727746e210f315138932e0aacb11_13637887.eps')
        #plt.tight_layout()
        #plt.savefig(res)

        return min_prob
        
    def model(self, k: int = 5, metric: str = 'size'):
        """Build optmization model with objective and constraints.

        Parameters
        ----------
        k : int, optional
            Number of concepts to select, by default 5
        metric : str
            * 'size': Size of extent and intent
            * 'tf-idf': Sum of `tf-idf` scores for each object-attribute pair

        Returns
        -------
        ``pulp`` problem object.
        """        
        # Initialize variables to optimize according to metric
        ext_var, int_var = self.init_model_variables(metric=metric)

        # Find subset of 5 concepts that maximize variable
        prob = pulp.LpProblem("Best_concepts", LpMaximize)
        prob += pulp.lpSum([self.x[i] * (int_var[i] + ext_var[i]) for i in self.concepts_idx])
        prob += pulp.lpSum([self.x[i] for i in self.concepts_idx]) == k
        for i in self.concepts_idx:
            prob += self.x[i] * ext_var[i] <= 0.95
        
        return prob

    def solve(self, k: int = 5, metric: str = 'size', solver: object = PULP_CBC_CMD, msg: bool = False) -> list:
        """Solve optimization problem (default solver is CBC MILP).

        Parameters
        ----------
        k : int, optional
            Number of concepts to select, by default 5
        metric : str
            * 'size': Size of extent and intent
            * 'tf-idf': Sum of `tf-idf` scores for each object-attribute pair
        msg : bool, optional
            If `True`, print solver log in standard output, by default False

        Returns
        -------
        list
            List of concepts maximizing optimization problem.
        """        
        
        # Solve optimization
        if metric in ('size'):
            prob = self.multi_obj_model(k=k, metric=metric)
        elif metric == 'tf-idf':
            prob = self.model(k=k, metric=metric)
            prob.solve(solver(msg=msg))
        
        concept_idxs = np.array([int(v.name.split('_')[1]) for v in prob.variables() if v.varValue > 0])
        scores = np.array([v.dj for v in prob.variables()])

        return list(self.concepts[concept_idxs]), scores
