from scipy import sparse
from src.graph import BipartiteGraph

from sknetwork.data.parse import from_edge_list


class FormalContext:
    def __init__(self, graph: BipartiteGraph):
        bunch = self._get_ohe_attributes(graph.node_attr['right'])
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

    def _deriv_operator(self, idx: int, type: str = 'ext') -> list:
        """Return derivative operator, i.e. extension or intention, given the index of an attribute, resp. object.

        Parameters
        ----------
        idx : int
            Index of attribute/object
        type : str, optional
            Type of derivative operator, by default 'ext'. Can be either:
                * `ext`: extention
                * `int`: intention

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
        res = list(names[matrix.indices[matrix.indptr[idx]:matrix.indptr[idx + 1]]])
        
        return res

    def extension(self, attributes: list) -> list:
        """Return maximal set of objects which share ``attributes``.

        Parameters
        ----------
        attributes : list
            Names of attributes (subset of ``self.M``)

        Returns
        -------
        list
            List of objects (subset of ``self.G``)
        """        
        self._build_attr_indx()
        
        ext_is = set()
        for attr in attributes:
            self._check_attr(attr)
            ext_i = set(self._deriv_operator(self.M2idx.get(attr), type='ext'))
            if len(ext_is) == 0:
                ext_is.update(ext_i)
            else:
                ext_is &= ext_i
        
        return list(ext_is)

    def intention(self, objects: list) -> list:
        """Return maximal set of attributes which share ``objects``.

        Parameters
        ----------
        objects : list
            Names of objects (subset of ``self.G``)

        Returns
        -------
        list
            List of attributes (subset of ``self.M``)
        """        
        self._build_obj_indx()

        int_is = set()
        for obj in objects:
            self._check_obj(obj)
            int_i = set(self._deriv_operator(self.G2idx.get(obj), type='int'))
            if len(int_is) == 0:
                int_is.update(int_i)
            else:
                int_is &= int_i

        return list(int_is)

    def _check_attr(self, attr: str):
        if not attr in self.M2idx:
            raise TypeError(f'Attribute {attr} is not known.')
    
    def _check_obj(self, obj: str) -> bool:
        if not obj in self.G2idx:
            raise TypeError(f'Object {obj} is not known.')
