import os

import networkx as nx
import numpy as np
import pandas as pd


def load_graphs_to_dataframe(directory="graphs", file_prefix="adj_matrix"):
    """
    Loads all .npy adjacency matrix files from the specified directory,
    flattens each matrix, and returns a pandas DataFrame with each row
    representing a flattened adjacency matrix.
    """
    matrices = []
    files = [
        f
        for f in os.listdir(directory)
        if f.startswith(file_prefix) and f.endswith(".npy")
    ]

    for file in sorted(files):
        file_path = os.path.join(directory, file)
        matrix = np.load(file_path)
        flattened_matrix = matrix.flatten()
        matrices.append(flattened_matrix)

    df = pd.DataFrame(matrices)
    return df


class RandomGraphGenerator:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def generate_random_graph(self):
        for i in range(1000):
            graph = nx.erdos_renyi_graph(self.num_nodes, 0.3)
            if nx.is_connected(graph):
                break
            # seed += 1
        return graph

    def generate_random_regular_graph(self, d=3):
        """
        Generates a random graph based on the number of nodes.
        Ensures that the graph is connected and undirected.
        """
        graph = nx.random_regular_graph(d=d, n=self.num_nodes)
        return graph

    def get_adjacency_matrix(self, graph):
        """
        Returns the adjacency matrix of the graph.
        """
        return nx.to_numpy_array(graph)

    def generate_multiple_graphs(self, shots):
        """
        Generates multiple random graphs and returns their adjacency matrices.
        """
        matrices = []
        for _ in range(shots):
            graph = self.generate_random_graph()
            matrix = self.get_adjacency_matrix(graph)
            matrices.append(matrix)
        return matrices

    def save_adjacency_matrices(self, matrices, file_prefix="adj_matrix"):
        """
        Saves the adjacency matrices to files.
        """
        N = matrices[0].shape[0]
        os.makedirs("graphs", exist_ok=True)
        for i, matrix in enumerate(matrices):
            file_name = f"graphs/{file_prefix}_{N}_{i+1}.npy"
            np.save(file_name, matrix)
            print(f"Adjacency matrix saved to {file_name}")
