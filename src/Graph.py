from __future__ import annotations
import copy
import json
import os


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
        return copy.deepcopy(self)

    def remove_edges_from(self, edge_subset: list):
        """Remove all edges specified in edge_subset.

        Parameters
        ----------
        edge_subset : list
            Edges to remove.
        """        
        self.E = list(set(self.E).difference(set(edge_subset)))

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
