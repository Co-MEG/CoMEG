from scipy import sparse
import numpy as np
import pandas as pd
import os
from typing import Optional, Union, TYPE_CHECKING

from src.graph import BipartiteGraph
from src.concept_lattice import ConceptLattice
from src.utils import get_oserror_dir

from sknetwork.data.parse import from_edge_list
from sknetwork.data import from_csv


class FormalContext:
    def __init__(self, graph: Union[BipartiteGraph, dict]):
        if isinstance(graph, BipartiteGraph):
            bunch = self._get_ohe_attributes(graph.node_attr['right'])
        elif isinstance(graph, dict):
            # for "from_csv" usage
            bunch = graph
        self.graph = graph
        self.G = bunch.names_row
        self.M = bunch.names_col
        self.I = bunch.biadjacency
        self.M2idx = {}
        self.G2idx = {}

    def _get_ohe_attributes(self, attributes: dict) -> dict:
        """Get set of attributes for Formal Context. For this purpose, original attributes are transformed into their
        one-hot-encoded form.

        Parameters
        ----------
        attributes : dict
            Dictionary of attributes with name of original attribute as key.

        Returns
        -------
        dict
            Bunch object in the form of a dictionary, containing:
                * `biadjacency`: Biadajcency matrix in csr format
                * `names_row`: List of names for rows
                * `names_col`: List of names for columns, i.e. one-hot-encoded attributes
        """        

        edge_list = []

        for k, attributes in attributes.items():
            for attr, value in attributes.items():
                if not isinstance(value, (list, dict)):
                    ohe_attr = f'{attr}_{value}'
                    edge_list.append((k, ohe_attr))

        bunch = from_edge_list(edge_list, bipartite=True, reindex=True)

        return bunch

    def _build_attr_indx(self):
        """Build attribute to index mapping.
        """        
        if len(self.M2idx) == 0:
            self.M2idx = {attr: idx for idx, attr in enumerate(self.M)}

    def _build_obj_indx(self):
        """Build object to index mapping.
        """        
        if len(self.G2idx) == 0:
            self.G2idx = {obj: idx for idx, obj in enumerate(self.G)}

    def _deriv_operator(self, idx: int, type: str = 'ext', return_names: bool = True) -> list:
        """Return derivative operator, i.e. extension or intention, given the index of an attribute, resp. object.

        Parameters
        ----------
        idx : int
            Index of attribute/object
        type : str, optional
            Type of derivative operator, by default 'ext'. Can be either:
                * `ext`: extention
                * `int`: intention
        return_names: bool
            If ``True`` return names of objects/attributes. Else return indexes, ``True`` by default.

        Returns
        -------
        list
            List of objects/attributes
        """       
        if type == 'ext':
            matrix = sparse.csr_matrix(self.I.T)
            names = self.G
        elif type == 'int':
            matrix = self.I
            names = self.M

        res = matrix.indices[matrix.indptr[idx]:matrix.indptr[idx + 1]]
        if return_names:
            res = list(names[res])
        
        return res

    def extension(self, attributes: Optional[list] = None, return_names: bool = True) -> list:
        """Return maximal set of objects which share ``attributes``. Without parameter, return the whole set 
        of objects.

        Parameters
        ----------
        attributes : list
            Names of attributes (subset of ``self.M``)
        return_names: bool
            If ``True`` return names of objects/attributes. Else return indexes, ``True`` by default.

        Returns
        -------
        list
            List of objects (subset of ``self.G``)
        """        
        self._build_attr_indx()
        
        ext_is = set()
        if (isinstance(attributes, list) and len(attributes) == 0) or attributes is None:
            ext_is = self.G
        else:
            for attr in attributes:
                self._check_attr(attr)
                ext_i = set(self._deriv_operator(self.M2idx.get(attr), type='ext', return_names=return_names))
                if len(ext_is) == 0:
                    ext_is.update(ext_i)
                else:
                    ext_is &= ext_i
            
        return list(ext_is)

    def intention(self, objects: Optional[list] = None, return_names: bool = True) -> list:
        """Return maximal set of attributes which share ``objects``. Without parameter, return the whole set
        of attributes.

        Parameters
        ----------
        objects : list
            Names of objects (subset of ``self.G``)
        return_names: bool
            If ``True`` return names of objects/attributes. Else return indexes, ``True`` by default.

        Returns
        -------
        list
            List of attributes (subset of ``self.M``)
        """        
        self._build_obj_indx()

        int_is = set()
        if (isinstance(objects, list) and len(objects) == 0) or objects is None:
            int_is = self.M
        else:
            for obj in objects:
                self._check_obj(obj)
                int_i = set(self._deriv_operator(self.G2idx.get(obj), type='int', return_names=return_names))
                if len(int_is) == 0:
                    int_is.update(int_i)
                else:
                    int_is &= int_i

        return list(int_is)

    def lattice(self, algo: str = 'in-close') -> ConceptLattice:
        """Return Concept Lattice of the Formal Context.

        Parameters
        ----------
        algo : str, optional
            Algorithm name, by default 'in-close' (in-close). Can be either:
                * 'in-close': In Close 
                * 'CbO': Close by One

        Returns
        -------
        ConceptLattice
            ``ConceptLattice`` instance.
        """        
        self._build_attr_indx()
        self._build_obj_indx()
        return ConceptLattice.from_context(self, algo=algo)

    def _check_attr(self, attr: str):
        if not attr in self.M2idx:
            raise TypeError(f'Attribute {attr} is not known.')
    
    def _check_obj(self, obj: str) -> bool:
        if not obj in self.G2idx:
            raise TypeError(f'Object {obj} is not known.')

    @staticmethod
    def from_csv(path: str) -> 'FormalContext':
        """Build ``FormalContext`` from csv file.

        Parameters
        ----------
        path : _type_
            Path to `.csv` file

        Returns
        -------
        FormalContext
            Formal context
        """        
        bunch = from_csv(path, bipartite=True, weighted=False)
        context = FormalContext(bunch)
        
        return context

    def to_csv(self, path: str, sep: str = ','):
        """Save formal context interactions to `.csv` file.

        Parameters
        ----------
        path : str
            Path to `csv.file`
        sep : str
            Separator
        """        
        mat = np.hstack((self.G.reshape(-1, 1), self.I.todense()))
        cols = np.concatenate([np.array(['names']), self.M])
        df = pd.DataFrame(mat, columns=cols)
        
        try:
            df.to_csv(path, index=False, sep=sep)
        except OSError as e:
            os.mkdir(get_oserror_dir(e))
            df.to_csv(path, index=False, sep=sep)
        print(f'Saved!')

    def to_concept_csv(self, path: str):
        """Save formal context interactions to `concepts` library `.csv` format. 

        Parameters
        ----------
        path : str
            Path to `csv.file`
        """        
        # Convert FormalContext format
        I_str = self.I.todense().astype(int).astype(str)
        I_str[I_str == '1'] = 'X'
        I_str[I_str == '0'] = ''

        # Build row and column names
        row_names = self.G.reshape(-1, 1)
        col_names = np.concatenate([np.array(['names']), self.M])

        # Concatenate and save
        mat = np.hstack((row_names, I_str))
        df = pd.DataFrame(mat, columns=col_names)
        
        try:
            df.to_csv(path, index=False)
        except OSError as e:
            os.mkdir(get_oserror_dir(e))
            df.to_csv(path, index=False)
        
        # Add necessary line at end of file
        separators = ',' * (len(col_names) - 1)
        with open(path, 'a') as f:
            f.write("R = X = {}" + separators)
        print(f'Saved!')
