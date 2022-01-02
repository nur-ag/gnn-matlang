import torch
from torch_geometric.data import DataLoader, Batch

import numpy as np
import igraph as ig

import sys
sys.path.append('../IGEL/src')
sys.path.append('../IGEL/gnnml-comparisons')


class IGELPreprocessor:
    '''A preprocessor to include IGEL (neighbourhood structure) node/edge features.

    If the `model` parameter is None, a model is trained during initialization.
    '''
    def __init__(self, 
            seed=0, 
            distance=1, 
            vector_length=5,
            node_feature_fields=['x'], 
            edge_feature_map={'edge_index': 'edge_attr', 'edge_index2': 'edge_attr2'},
            edge_fn=lambda x, y: x * y):
        self.distance = distance
        self.seed = seed
        self.vector_length = vector_length
        self.node_feature_fields = node_feature_fields
        self.edge_feature_map = edge_feature_map
        self.edge_fn = edge_fn

    def __call__(self, data):
        if self.distance < 1:
            return data

        # We have something to preprocess since distance is nonzero
        data = Batch.from_data_list(data)
        G = self.global_graph(data)
        model = self.train_igel_model(G)

        # Detach the embeddings so that we don't backprop at all -- harder baseline!
        embeddings = torch.Tensor(model(G.vs, G).cpu().detach().numpy())


        # Add node features where required
        for field in self.node_feature_fields:
            base_features = getattr(data, field, None)
            if base_features is None:
                continue
            concat_result = torch.cat([base_features, embeddings], axis=-1)
            setattr(data, field, concat_result)

        # Add edge features where required
        if self.edge_feature_map:
            for index_field, feature_field in self.edge_feature_map.items():
                edge_indices = getattr(data, index_field, None)
                edge_attr = getattr(data, feature_field, None)
                if edge_indices is None or edge_attr is None:
                    continue

                edge_pair_emb = embeddings[edge_indices]
                edge_emb = self.edge_fn(edge_pair_emb[0], edge_pair_emb[1])
                concat_result = torch.cat([edge_attr, edge_emb], axis=-1)
                setattr(data, feature_field, concat_result)
        return data.to_data_list()

    def global_graph(self, data):
        edges = data.edge_index2

        # Iterate through edge lists to collect all edges
        edges_numpy = edges.numpy()
        edge_tuples = list(zip(edges_numpy[0], edges_numpy[1]))
        graph_edges = []
        current_graph = []
        for edge in edge_tuples:
            a, b = edge
            if not current_graph:
                current_graph.append(edge)
                continue
            if a < current_graph[-1][0]:
                graph_edges.append(current_graph)
                current_graph = [edge]
                continue
            current_graph.append(edge)
        if current_graph:
            graph_edges.append(current_graph)

        # We now have a list of graph edges for each graph -- shift them and compute total nodes
        graph_nodes = [set([node for edge in edges for node in edge]) for edges in graph_edges]
        graph_sizes = [len(nodes) for nodes in graph_nodes]
        graph_sizes_cumsum = np.cumsum([0] + graph_sizes)
        total_nodes = graph_sizes_cumsum[-1]
        graph_sizes_shift = graph_sizes_cumsum[:-1]
        graph_shifted_edges = [tuple([node + shift for node in edge]) 
                               for edges, shift in zip(graph_edges, graph_sizes_shift) 
                               for edge in edges]

        # The global graph is the graph with all the nodes and edges
        G = ig.Graph()
        G.add_vertices(total_nodes)
        G.add_edges(graph_shifted_edges)
        G.vs['name'] = [str(n.index) for n in G.vs]
        return G

    def train_igel_model(self, G):
        from igel_embedder import get_unsupervised_embedder
        return get_unsupervised_embedder(G, 
                                         self.distance, 
                                         self.seed,
                                         self.vector_length)


class AddLabelTransform:
    '''A simple transform to add the label to all nodes in the graph.

    Meant to be used for debugging, as a model with the label as a feature should get 100% accuracy.
    '''
    def __call__(self, data):
        x_indices = data.edge_index[0].tolist()
        current_graph_index = 0
        previous_node_index = x_indices[0]
        graph_indices = []
        for index in x_indices:
            if index < previous_node_index:
                current_graph_index += 1
            previous_node_index = index
            graph_indices.append(current_graph_index)
        label_values = data.y[graph_indices]
        label_values = label_values[:data.x.shape[0]]
        data.x = torch.cat([data.x, label_values.reshape(-1, 1)], axis=-1)
        return data


class LambdaReduceTransform:
    '''A transform implementing a reduce over transform functions.

    The output of this function is the output of running the transforms in order.
    '''
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, data):
        output = data
        for transform_fn in self.transforms:
            output = transform_fn(output)
        return output
