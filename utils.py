import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graph(graph):
    """
    Plots the graph.
    """
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10)
    plt.show(block=True)
    
    
def load_graph_from_npy(file_path):
    """
    Load a graph from a .npy file containing an adjacency matrix.

    Parameters:
    - file_path: str, path to the .npy file containing the adjacency matrix.

    Returns:
    - G: networkx.Graph, the graph created from the adjacency matrix.
    """
    adj_matrix = np.load(file_path)

    G = nx.from_numpy_array(adj_matrix)

    return G