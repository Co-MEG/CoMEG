import json
import numpy as np
from tqdm import tqdm

from data import load_data


def dataset2json(dataset: str, outpath: str):
    """Save sknetwork dataset to SIAS-Miner JSON format.
    
    Parameters
    ----------
    dataset: str
        Dataset name
    outpath: str
        Output path
    """
        
    adjacency, biadjacency, names, names_col, labels = load_data(dataset)
    n = adjacency.shape[0] 
    m = biadjacency.shape[1]
    out_graph = {}

    # Dataset name
    out_graph['descriptorName'] = dataset

    # Attribute names
    out_graph['attributeName'] = [str(i) for i in range(m)]

    # Edges
    edges = [{'vertexId': str(u), 'connected_vertices': list((adjacency[u].indices).astype(str))} for u in range(n)]
    out_graph['edges'] = edges
    print(f'Edges done!')

    # Vertices: nodes + attributes
    vertices = []
    for u in tqdm(range(n)):
        tmp = {'vertexId': str(u)}
        feats = np.zeros(m)
        feats[biadjacency[u].indices] = biadjacency[u].data
        tmp['descriptorsValues'] = list(map(int, feats))
        vertices.append(tmp)
    out_graph['vertices'] = vertices
    print(f'Verttices done!')

    # Save data
    print(f'Saving data...')
    with open(outpath, 'w') as f:
        json.dump(out_graph, f)


# ==================================================================
if __name__=='__main__':
    #datasets = ['wikivitals']
    datasets = ['wikivitals-fr', 'wikischools']
    for dataset in datasets:
        dataset2json(dataset, f'data/graph_{dataset}.json')
        print(f'{dataset} done!')
