import random

from src.Graph import BipartiteGraph

# Data path
IN_PATH = 'goodreads_poetry'
USE_CACHE = True
TEST_SIZE = 0.3

if __name__ == '__main__':
    
    # Build graph
    g = BipartiteGraph()
    g.load_data(IN_PATH, use_cache=USE_CACHE)

    # Verif
    print(type(g))
    print(f"Number of users: {len(g.V.get('left'))}")
    print(f"Number of books: {len(g.V.get('right'))}")
    print(f"Number of reviews: {len(g.E)}")

    # Link prediction
    #train_g = g.copy()
    #print(type(train_g))
    #print(f"Number of users: {len(train_g.V.get('left'))}")
    #print(f"Number of books: {len(train_g.V.get('right'))}")
    #print(f"Number of reviews: {len(train_g.E)}")
    m = g.number_of_edges()
    print(len(g.E))
    print(len(g.edge_attr))

    train_g, test_g = g.train_test_split(test_size=TEST_SIZE)
    
    print(f"Number of users in train-test: {len(train_g.V.get('left'))}-{len(test_g.V.get('left'))}")
    print(f"Number of books in train-test: {len(train_g.V.get('right'))}-{len(test_g.V.get('right'))}")
    print(f"Number of reviews in train-test: {len(train_g.E)}-{len(test_g.E)}")
    print(f"Number of edge attributes in train-test: {len(train_g.edge_attr)}-{len(test_g.edge_attr)}")

    #g_test = BipartiteGraph()
    #g_test.add_edges_from(edge_subset)


    