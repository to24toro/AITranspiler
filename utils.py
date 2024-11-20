import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx


def plot_graph(graph):
    """
    Plots the graph.
    """
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=10,
    )
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


def convert_to_rx(g):
    rx_graph = rx.PyGraph()
    rx_graph.add_nodes_from(list(g.nodes()))
    rx_graph.add_edges_from([(u, v, None) for u, v in g.edges()])
    return rx_graph
