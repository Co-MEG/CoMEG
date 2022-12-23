from __future__ import annotations

import copy
import json
import numpy as np
import os
import random
from scipy import sparse
from sknetwork.data.parse import from_edge_list
from sknetwork.ranking import PageRank, top_k
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
        self.degrees = {}
        self.neighbors_2hops = {}
        self.neighbors = {}

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
            print('Loading data...')
            os.makedirs(in_path_preproc, exist_ok=True)
            files = os.listdir(in_path_raw)

            for f in files:
                if f.endswith('json'):
                    in_path_file = os.path.join(in_path_raw, f)
                    file = open(in_path_file)
                    
                    # Book attributes
                    if 'books' in f:
                        print(f'   Loading book information...')
                        books = {}
                        excluded_attributes = ['popular_shelves', 'link', 'url', 'image_url', \
                                                'book_id', 'isbn13', 'isbn', 'work_id', 'text_reviews_count', 'asin', \
                                                'kindle_asin', 'average_rating', 'ratings_count', 'num_pages', \
                                                'publication_day']

                        for book in file:
                            data = json.loads(book)
                            books[data['book_id']] = {attr: val for attr, val in data.items() \
                                if attr not in excluded_attributes}

                    # Interaction information
                    elif 'interactions' in f:
                        print(f'   Loading interactions information...')
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
            self._build_csr()

            # Save json
            self._save_json(out_path)
            print(f"Saved data to '{out_path}'.")

        elif os.path.isdir(in_path_preproc):
            self._load_json(out_path)
            self._build_csr()
            print(f"Loaded data from '{out_path}'.")

        else:
            raise FileExistsError(f"'use_cache is set to True but {in_path_preproc}' doest not exists.")

    def _build_csr(self):
        """Build csr adjacency matrix.
        """        
        bunch = from_edge_list(self.E, bipartite=True)
        self.adjacency_csr = bunch.biadjacency
        self.names_row = bunch.names_row
        self.names_col = bunch.names_col
        self.label2idx_rows = {lab: idx for idx, lab in enumerate(self.names_row)}
        self.label2idx_cols = {lab: idx for idx, lab in enumerate(self.names_col)}

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
        self.adjacency_csr.data[mask] = 0
        self.adjacency_csr.eliminate_zeros()
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

    def size(self) -> int:
        """ Return the size of the graph, i.e. |V| + |E|.
        
        Returns
        -------
        int
            Size of the graph.
        """
        return self.number_of_edges() + self.number_of_nodes()
        
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
        List
            List of neigbors for target node.
        """                
        if transpose:
            matrix = sparse.csr_matrix(self.adjacency_csr.T)
            idx = self.label2idx_cols.get(node)
            #idx = np.where(self.names_col == node)[0][0]
            names = self.names_row
        else:
            matrix = self.adjacency_csr
            idx = self.label2idx_rows.get(node)
            #idx = np.where(self.names_row == node)[0][0]
            names = self.names_col

        #if node in self.neighbors:
        #    neighbors = self.neighbors.get(node)
        #else:
        neighbors = list(names[matrix.indices[matrix.indptr[idx]:matrix.indptr[idx + 1]]])
        if node not in self.degrees:
            self.degrees[node] = len(neighbors) # save degree of node
        
        return neighbors
        
    def get_neighbors_2hops(self, node: str, transpose: bool = False) -> list:
        """Get 2-hop neighbors of a node.

        Parameters
        ----------
        node : str
            Target node
        transpose : bool, optional
            If `True`, transpose the adjacency matrix, by default False

        Returns
        -------
        List
            List of 2-hops neighbors
        """        
        n_2hops = []
        if node in self.neighbors_2hops:
            n_2hops = self.neighbors_2hops.get(node)
        else:    
            neighbors_1hop = self.get_neighbors(node, transpose)
            for n in neighbors_1hop:
                n_2hops += self.get_neighbors(n, ~transpose)
            #self.neighbors_2hops[node] = n_2hops
        
        return n_2hops

    def has_edge(self, u: str, v: str) -> bool:
        """Return `True` if (u, v) is an edge in the graph.

        Parameters
        ----------
        u : str
            Source node
        v : str
            Destination node

        Returns
        -------
        bool
            `True` if (u, v) is an edge.
        """        
        try:
            neighbors = self.get_neighbors(u)
        except IndexError:
            print(f'u, v: {u}, {v}')
        return v in neighbors

    def subgraph_vicinity_degree(self, edge: Tuple, min_degree_left: int = 2, min_degree_right: int = 3) -> BipartiteGraph:
        subgraph = BipartiteGraph()
        src, dst = edge
        u_idx = self.label2idx_rows.get(src)
        v_idx = self.label2idx_cols.get(dst)

        # Apply filter on node degrees
        degrees_left = self.adjacency_csr.dot(np.ones(self.adjacency_csr.shape[1])) >= min_degree_left
        degrees_right = self.adjacency_csr.T.dot(np.ones(self.adjacency_csr.shape[0])) >= min_degree_right

        # 2-hops left nodes
        neighbs_idx = self.adjacency_csr.indices[self.adjacency_csr.indptr[u_idx]:self.adjacency_csr.indptr[u_idx + 1]]
        nnz_idx = np.flatnonzero(neighbs_idx * degrees_right[neighbs_idx])
        n_u = neighbs_idx[nnz_idx]
        edges_u = [(src, self.names_col[x]) for x in n_u]

        edges_2hops_u = []
        for v in n_u:
            neighbs_idx = self.adjacency_csr.T.indices[self.adjacency_csr.T.indptr[v]:self.adjacency_csr.T.indptr[v+1]]
            nnz_idx = np.flatnonzero(neighbs_idx * degrees_left[neighbs_idx])
            n_2hops_u = neighbs_idx[nnz_idx]
            edges_2hops_u.append([(u, v) for u in n_2hops_u])
        
        edges_2hops_u_rav = [(self.names_row[x[0]], self.names_col[x[1]]) for sublist in edges_2hops_u for x in sublist]

        # 2-hops right nodes
        neighbs_idx = self.adjacency_csr.T.indices[self.adjacency_csr.T.indptr[v_idx]:self.adjacency_csr.T.indptr[v_idx + 1]]
        nnz_idx = np.flatnonzero(neighbs_idx * degrees_left[neighbs_idx])
        n_v = neighbs_idx[nnz_idx]
        edges_v = [(self.names_row[x], dst) for x in n_v]

        edges_2hops_v = []
        for u in n_v:
            neighbs_idx = self.adjacency_csr.indices[self.adjacency_csr.indptr[u]:self.adjacency_csr.indptr[u+1]]
            nnz_idx = np.flatnonzero(neighbs_idx * degrees_right[neighbs_idx])
            n_2hops_v = neighbs_idx[nnz_idx]
            edges_2hops_v.append([(u, v) for v in n_2hops_v])
        
        edges_2hops_v_rav = [(self.names_row[x[0]], self.names_col[x[1]]) for sublist in edges_2hops_v for x in sublist]

        edges = list(set([(src, dst)])
                    .union(set(edges_u))
                    .union(set(edges_v))
                    .union(set(edges_2hops_u_rav))
                    .union(set(edges_2hops_v_rav))
                )
        u_list, v_list = zip(*edges)
        subgraph.V['left'] = list(set(u_list))
        subgraph.V['right'] = list(set(v_list))
        subgraph.node_attr['right'] = {x: self.node_attr['right'].get(x) for x in subgraph.V['right']}
        subgraph.E = edges

        subgraph._build_csr()
        
        return subgraph

    def subgraph_vicinity(self, edge: Tuple, method: str = 'ppr') -> BipartiteGraph:
        """Build subgraph in the vicinity of an edge (u, v). The subgraph is made of the edge itself, as well as all 
        the edges (x, y) such that x and y are in the sets of nodes obtained using 'method'.

        Parameters
        ----------
        edge : Tuple
            Edge on which subgraph is built.
        method : str
            Method used to extract nodes in the vicinity of the prediction. Either:
                * 'ppr': Personalized PageRank with predicted nodes as seeds
                * 'neighbors': 1-hop neighbors of predicted nodes 

        Returns
        -------
        BipartiteGraph
            Subgraph 
        """        
        subgraph = BipartiteGraph()
        u, v = edge[0], edge[1]

        if method == 'neighbors':
            # Get subgraph nodes and node attributes
            n_u = self.get_neighbors(u, transpose=False)
            n_v = self.get_neighbors(v, transpose=True)
            subgraph.V['right'] = list(set(n_u).union(set([v])))
            subgraph.V['left'] = list(set(n_v).union(set([u])))
            subgraph.node_attr['right'] = {x: self.node_attr['right'].get(x) for x in subgraph.V['right']}
            
            # Get subgraph edges and edge attributes
            edges_u = set([(u, y) for y in n_u])
            edges_v = set([(x, v) for x in n_v])
            edges = set([(x, y) for y in n_u for x in n_v if self.adjacency_csr[self.label2idx_rows.get(x), self.label2idx_cols.get(y)] > 0])
            subgraph.E = list(edges_u.union(edges_v).union(edges).union(set([(u, v)])))

        elif method == 'ppr':
            k = 15 # number of nodes to select from each personalized PageRank
            pagerank = PageRank()
            seed_row = {self.label2idx_rows.get(u): 1}
            seed_col = {self.label2idx_cols.get(v): 1}
            
            # Ppr with seed on left node
            pagerank.fit_transform(self.adjacency_csr, seeds_row=seed_row)
            top_row_left = top_k(pagerank.scores_row_, k)
            top_col_left = top_k(pagerank.scores_col_, k)
            # Ppr with seed on right node
            pagerank.fit_transform(self.adjacency_csr, seeds_col=seed_col)
            top_row_right = top_k(pagerank.scores_row_, k)
            top_col_right = top_k(pagerank.scores_col_, k)
            # Union of selected nodes on left and right
            n_u = set(top_row_left).union(set(top_row_right))
            n_v = set(top_col_left).union(set(top_col_right))
            
            edges = set([(self.names_row[x], self.names_col[y]) for x in n_u for y in n_v if self.adjacency_csr[x, y] > 0])
            subgraph.E = list(edges)

            n_left, n_right = zip(*(edges))
            subgraph.V['left'] = list(set(n_left))
            subgraph.V['right'] = list(set(n_right))
            subgraph.node_attr['right'] = {x: self.node_attr['right'].get(x) for x in subgraph.V['right']}

        # Build adjacency matrix of the subgraph
        subgraph._build_csr()

        return subgraph
