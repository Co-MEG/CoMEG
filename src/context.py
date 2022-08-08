from src.graph import BipartiteGraph

from sknetwork.data.parse import from_edge_list


class FormalContext:
    def __init__(self, graph: BipartiteGraph):
        bunch = self._get_ohe_attributes(graph.node_attr['right'])
        self.G = bunch.names_row
        self.M = bunch.names_col
        self.I = bunch.biadjacency

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
        
        
        
