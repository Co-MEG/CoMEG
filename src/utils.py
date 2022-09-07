# Utility functions
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics
from typing import Union

import networkx as nx

from src.concept_lattice import ConceptLattice

def get_oserror_dir(e: OSError) -> str:
    """Return non-existent directory name raised within OSError. """    
    return str(e).split(': ')[1].strip("''")


def plot_roc_auc(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], ax, label: str = 'Adamic-Adar'):
    """Plot ROC-AUC curve.

    Parameters
    ----------
    y_true : Union
        Ground truth values
    y_pred : Union
        Predicted values
    ax : _type_
        Axe
    label : str
        Plot label
    """    
    fpr, tpr, thresholds = metrics.roc_curve(y_true,  y_pred) 
    auc = metrics.roc_auc_score(y_true, y_pred)

    ax.plot(fpr, tpr, label='Adamic Adar')
    ax.set_title(f'ROC AUC Curve - AUC = {auc:.4f}', weight='bold')
    ax.legend()

def load_predictions(path: str) -> dict:
    """Load edge predictions and return result in dictionary.

    Parameters
    ----------
    path : str
        Path to edge predictions

    Returns
    -------
    Dictionary of edge predictions
    """    
    # Load predictions
    D = {}

    # Convert list of dicts into one big dictionary of values
    print(f'Load result values...')
    with open(path) as f:
        lines = f.read().splitlines()
        for l in lines:            
            pair = l.strip('{}').split(': ')
            k = pair[0]
            v = pair[1]
            vals = []
            for idx, i in enumerate(v.strip('[]').split(', ')):
                if idx <= 1:
                    vals.append(i.strip('""'))
                else:
                    vals.append(float(i))
            D.update({k: vals})

    return D

def draw_bipartite_graph(G: nx.classes.graph.Graph, top_nodes, color_map, ax):
    """Draw networkX bipartite graph

    Parameters
    ----------
    G : nx.classes.graph.Graph
        NetworkX bipartite graph
    ax : _type_
        Axe
    """    
    nx.draw_networkx(
        G,
        pos = nx.drawing.layout.bipartite_layout(G, top_nodes),
        edge_color = [G[u][v]['color'] for u, v in G.edges()],
        node_color = color_map,
        width = [G[u][v]['weight'] for u, v in G.edges()],
        ax = ax
    )

def render_title(title: list, k: int = 5) -> str:
    """Format elements in list into a title displayed on :math:`k` lines.

    Parameters
    ----------
    title : list
        List of strings
    k : int, optional
        Number of lines on which display title, by default 3

    Returns
    -------
    str
        String containing title displayed on k lines.
    """    
    res = ''
    i = 0
    for elem in title:
        i += 1
        if i >= k:
            res += elem + ', \n'
            i = 0
        else:
            res += elem + ', '
    
    return res

def draw_pairwise_matrix(mat: np.ndarray, lattice: ConceptLattice, ax, path: str = None,
                        cmap: matplotlib.colors.LinearSegmentedColormap = plt.cm.Blues, show: bool = True):
    """Draw pairwise distance matrix.

    Parameters
    ----------
    mat : np.ndarray
        Matrix of pairwise distances.
    lattice : ConceptLattice
        Concept lattice built upon a specific context.
    ax : _type_
        Matplotlib axe.
    path : str, optional
        Path to save image, by default None
    cmap : matplotlib.colors.LinearSegmentedColormap, optional
        Coloring parameter, by default plt.cm.Blues
    show : bool, optional
        If `True`, display result, by default True
    """                        
    # Draw pairwise distance matrix
    ax.matshow(mat, cmap=cmap)
    tick_marks = np.arange(len(lattice.concepts))

    plt.xticks(tick_marks, range(len(lattice.concepts)), fontsize=7)
    plt.yticks(tick_marks, [x[0] for x in lattice.concepts], fontsize=7)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            c = round(mat[j, i], 2)
            ax.text(i, j, str(c), va='center', ha='center')
    
    if show:
        plt.show()
    elif path != None:
        res = os.path.join(path)
        plt.tight_layout()
        plt.savefig(res)