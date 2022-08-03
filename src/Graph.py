from __future__ import annotations

import copy
import json
import numpy as np
import os
import random
from sknetwork.data.parse import from_edge_list
from typing import Tuple


class BipartiteGraph:
    """Bipartite graph class.

    Attributes
    ----------
    V : dict
        Nodes of the graph divided into `left` and `right` lists.
    E: list
        Edges of the graph. Each edge is a tuple (src, dest), with:
            * src : source node id
            * dst : destination node id
    node_attr : dict
        Attributes of each node, divided into `left` and `right` dictionaries.
    edge_attr: list
        Attributes of the edge in the form of a list of dictionaries, with same length as the number of edges.
    """    
    def __init__(self):        
        self.V = {'left': [], 'right': []}
        self.E = []
        self.node_attr = {'left': {}, 'right': {}}
        self.edge_attr = []
        self.adjacency_csr = None

    def load_data(self, path: str, use_cache: bool = True):
        """Load data and save information as json object.

        Parameters
        ----------
        path : str
            Path to input data. If `path` points to raw data, data is first preprocessed.
        use_cache : bool (`True`)
            Use cached data (when multiple runs).
        """        
        in_path = os.path.join(os.getcwd(), 'data', path)
        in_path_raw = os.path.join(in_path, 'raw')
        in_path_preproc = os.path.join(in_path, 'preproc')
        dirname = os.path.basename(path)
        out_path = os.path.join(in_path_preproc, dirname)
        
        if not use_cache:
            os.makedirs(in_path_preproc, exist_ok=True)
            files = os.listdir(in_path_raw)

            for f in files:
                if f.endswith('json'):
                    in_path_file = os.path.join(in_path_raw, f)
                    file = open(in_path_file)
                    
                    # Book attributes
                    if 'books' in f:
                        books = {}
                        excluded_attributes = ['popular_shelves', 'description', 'link', 'url', 'image_url']

                        for book in file:
                            data = json.loads(book)
                            books[data['book_id']] = {attr: val for attr, val in data.items() \
                                if attr not in excluded_attributes}
                    
                    # Interaction information
                    elif 'interactions' in f:
                        links = []
                        edge_attrs = []
                        user_ids = set()
                        book_ids = set()

                        for review in file:
                            data = json.loads(review)

                            u = data.get('user_id')
                            user_ids.add(u)
                            v = data.get('book_id')
                            book_ids.add(v)
                            edge_attr = {'is_read': data.get('is_read'), 'rating': data.get('rating')}
                            link = (u, v)
                            links.append(link)
                            edge_attrs.append(edge_attr)
            
            # Build JSON skeleton
            self.V = {'left': list(user_ids), 'right': list(book_ids)}
            self.E = links
            self.node_attr = {'left': [], 'right': books}
            self.edge_attr = edge_attrs

            # Save json
            self._save_json(out_path)
            print(f"Saved data to '{out_path}'.")
        elif os.path.isdir(in_path_preproc):
            self._load_json(out_path)
            print(f"Loaded data from '{out_path}'.")
        else:
            raise FileExistsError(f"'use_cache is set to True but {in_path_preproc}' doest not exists.")

    def _load_json(self, path: str):
        """Load data from preprocessed json file.

        Parameters
        ----------
        path : str
            Path to preprocessed json file.
        """        
        file = open(path + '.json')
        data = json.load(file)
        self.V = data['V']
        self.node_attr = data['node_attr']
        self.edge_attr = data['edge_attr']
        for e in data['E']:
            edge = (e[0], e[1])
            self.E.append(edge)

    def _save_json(self, path: str):
        """Save data to json.

        Parameters
        ----------
        path : str
            output path
        """        
        data = {'V': self.V, 'E': self.E, 'node_attr': self.node_attr, 'edge_attr': self.edge_attr}
        out_file = open(path + '.json', 'w')
        json.dump(data, out_file)

    def copy(self) -> BipartiteGraph:
        """Returns a deep copy of the BipartiteGraph object.

        Returns
        -------
        BipartiteGraph deep copy
        """        
        #TODO: Deepcopy needs to be faster
        return copy.deepcopy(self)

    def train_test_split(self, test_size: float = 0.3) -> Tuple:
        """Split graph into train and test subgraphs.

        Parameters
        ----------
        test_size : float, optional
            Proportion of edges to keep in test graph, by default 0.3

        Returns
        -------
        Tuple
            Tuple of `BipartiteGraph` corresponding to train and test subgraphs.
        """        
        m = self.number_of_edges()
        train_g = self.copy()
        test_g = self.copy()

        # Sample edge indexes
        edge_subset_idx = random.sample(range(m), k=int(test_size * m))
        
        # Mask
        mask = np.zeros(self.number_of_edges()).astype(bool)
        mask[edge_subset_idx] = True

        # Remove edges from train graph
        train_g.remove_edges_at(mask)
        
        # Create test graph with remaining edges
        test_g.remove_edges_at(~mask)

        return train_g, test_g
        

    def remove_edges_at(self, mask: np.ndarray):
        """Remove all edges according to mask. Edge attributes are removed as well.

        Parameters
        ----------
        mask : np.ndarray
            Mask with `True` values at index of edge to remove.
        """       
        # Filter edges and corresponding attributes
        self.E = list(np.array(self.E)[~mask])
        self.edge_attr = list(np.array(self.edge_attr)[~mask])

    def number_of_edges(self) -> int:
        """Return number of edges in the graph.

        Returns
        -------
        int
            Number of edges.
        """        
        return len(self.E)

    def number_of_nodes(self) -> int:
        """Return number of nodes in the graph.

        Returns
        -------
        int
            Total number of nodes.
        """        
        return len(self.V.get('left')) + len(self.V.get('right'))

    def get_neighbors(self, node: str, transpose: bool = False) -> list:
        """Get neighbors of a node.

        Parameters
        ----------
        node : str
            Target node
        transpose : bool, optional
            If `True`, transpose the adjacency matrix, by default False

        Returns
        -------
        list
            List of neigbors for target node.
        """        
        if self.adjacency_csr is None:
            bunch = from_edge_list(self.E, bipartite=True)
            self.adjacency_csr = bunch.biadjacency
            self.names_row = bunch.names_row
            self.names_col = bunch.names_col
        
        if transpose:
            matrix = self.adjacency_csr.T
            idx = np.where(self.names_col == node)[0][0]
            names = self.names_col
        else:
            matrix = self.adjacency_csr
            idx = np.where(self.names_row == node)[0][0]
            names = self.names_row

        return names[matrix.indices[matrix.indptr[idx]:matrix.indptr[idx + 1]]]
        