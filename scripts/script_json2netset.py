import json
import numpy as np
import pickle
from scipy import sparse

from sknetwork.data import Bunch, from_adjacency_list


def load_json_dataset(inpath, filename):
    with open(f'{inpath}/{filename}.json') as f:
        data = json.load(f)
    return data

if __name__=='__main__':
    
    inpath = 'data'
    filename = 'londonGraph'

    dataset = Bunch()
    meta = Bunch()
    nodelabel2idx = dict()

    # Load data
    data = load_json_dataset(inpath, filename)

    # Metadata
    meta.name = data.get('descriptorName')
    meta.description = ''
    meta.source = ''
    meta.date = 'February 2023'

    # Attributes information
    names = []
    names_col = np.asarray(data.get('attributesName'))
    vertices = data.get('vertices')

    biadjacency_dense = np.zeros((len(vertices), len(names_col)))
    print(f'Initial number of vertices: {len(vertices)}')

    for i, v in enumerate(vertices):
        nodelabel2idx[v.get('vertexId')] = i
        biadjacency_dense[i, :] = np.asarray(v.get('descriptorsValues'))
        names.append(v.get('vertexId'))

    biadjacency = sparse.csr_matrix(biadjacency_dense)
    print(f'Biadjacency shape: {biadjacency.shape}')

    # Vertex information
    edges = data.get('edges')
    adj_list = {} 
    for i, e in enumerate(edges):
        adj_list[nodelabel2idx.get(e.get('vertexId'))] = [nodelabel2idx.get(x) for x in e.get('connected_vertices')]

    adjacency = from_adjacency_list(adj_list, directed=False, reindex=False)
    print(f'Adjacency: {adjacency.shape}, #edges: {adjacency.nnz}')

    dataset.adjacency = adjacency
    dataset.names = np.asarray(names)
    dataset.biadjacency = biadjacency
    dataset.names_col = names_col
    dataset.meta = meta

    # Save dataset
    with open(f'{inpath}/{filename}', 'bw') as f:
        pickle.dump(dataset, f)