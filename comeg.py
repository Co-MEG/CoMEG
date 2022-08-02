from src.Graph import BipartiteGraph

IN_PATH = 'goodreads_poetry'

if __name__ == '__main__':
    
    g = BipartiteGraph()
    g.load_data(IN_PATH)

    # Verif
    print(type(g))
    print(f"Number of users: {len(g.V.get('left'))}")
    print(f"Number of books: {len(g.V.get('right'))}")
    print(f"Number of reviews: {len(g.E)}")
    