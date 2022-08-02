import json
import os

class BipartiteGraph:
    """Bipartite graph class.

    Attributes
    ----------
    V : dict
        Nodes of the graph divided into `left` and `right` lists.
    E: list
        Edges of the graph. Each edge is a dictionary containing:
            * `u` : source node id
            * `v` : destination node id
            * `attr` : attributes of the edge in the form of a dictionary
    A : dict
        Attributes of each node, divided into `left` and `right` dictionaries.
    """    
    def __init__(self):        
        self.V = {'left': [], 'right': []}
        self.E = []
        self.A = {'left': {}, 'right': {}}

    def load_data(self, path: str):
        """Load data and save information as json object.

        Parameters
        ----------
        path : str
            Path to input data. If `path` points to raw data, data is first preprocessed.
        """        
        in_path = os.path.join(os.getcwd(), 'data', path)
        in_path_raw = os.path.join(in_path, 'raw')
        in_path_preproc = os.path.join(in_path, 'preproc')
        dirname = os.path.basename(path)
        out_path = os.path.join(in_path_preproc, dirname)
        
        if not os.path.isdir(in_path_preproc):
            os.mkdir(in_path_preproc)
            files = os.listdir(in_path_raw)

            for f in files:
                if f.endswith('json'):
                    in_path_file = os.path.join(in_path_raw, f)
                    file = open(in_path_file)
                    
                    # Book attributes
                    if 'books' in f:
                        books = {}
                        excluded_attributes = ['popular_shelves', 'description', 'link', 'url', 'image_url']

                        for i, book in enumerate(file):
                            data = json.loads(book)
                            books[data['book_id']] = {attr: val for attr, val in data.items() if attr not in excluded_attributes}
                    
                    # Interaction information
                    elif 'interactions' in f:
                        links = []
                        user_ids = set()
                        book_ids = set()

                        for review in file:
                            data = json.loads(review)

                            u = data.get('user_id')
                            user_ids.add(u)
                            v = data.get('book_id')
                            book_ids.add(v)
                            attr = {'is_read': data.get('is_read'),
                                    'rating': data.get('rating')}
                            link = {'u': u, 'v': v, 'attr': attr}
                            links.append(link)
            
            # Build JSON skeleton
            self.V = {'left': list(user_ids), 'right': list(book_ids)}
            self.E = links
            self.A = {'left': [], 'right': books}

            # Save json
            self._save_json(out_path)
            print(f"Saved data to '{out_path}'.")
        else:
            self._load_json(out_path)
            print(f"Loaded data from '{out_path}'.")

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
        self.E = data['E']
        self.A = data['A']

    def _save_json(self, path: str):
        """Save data to json.

        Parameters
        ----------
        path : str
            output path
        """        
        data = {'V': self.V, 'E': self.E, 'A': self.A}
        out_file = open(path + '.json', 'w')
        json.dump(data, out_file)
