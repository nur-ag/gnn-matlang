import torch
from torch_geometric.data import DataLoader, Batch

import numpy as np
import igraph as ig

import sys
sys.path.append('../IGEL/src')
sys.path.append('../IGEL/gnnml-comparisons')


class IGELPreprocessor:
    '''A preprocessor to include IGEL (neighbourhood structure) node/edge features.

    Args:
        seed (int): random seed to pass to the unsupervised IGEL training job.
        distance (int): IGEL encoding distance. If set to less than 1, this preprocessor is a no-op.
        vector_length (int): length of the learned unsupervised IGEL embeddings. If set to a negative number, use IGEL structural encoding features (no training).
        node_feature_fields (list[string]): list of names for fields containing node features.
        edge_feature_map (dict[string, string]): dictionary mapping edge index fields to edge attribute fields.
        edge_fn (function(tensor, tensor) :-> tensor): a function applied to node features when computing edge feature vectors. 
    '''
    def __init__(self, 
            seed=0, 
            distance=1, 
            vector_length=5,
            use_relative_degrees=False,
            node_feature_fields=['x'], 
            edge_feature_map={'edge_index': 'edge_attr', 'edge_index2': 'edge_attr2'},
            edge_fn=lambda x, y: x * y):
        self.distance = distance
        self.seed = seed
        self.vector_length = vector_length
        self.use_encoding = vector_length < 0
        self.use_relative_degrees = use_relative_degrees
        self.node_feature_fields = node_feature_fields
        self.edge_feature_map = edge_feature_map
        self.edge_fn = edge_fn

    def __call__(self, data, training_data=None):
        if self.distance < 1:
            return data

        if training_data is None:
            training_data = data

        # We have something to preprocess since distance is nonzero
        data_list = data
        training_data_list = Batch.from_data_list(training_data)
        G = self.global_graph(training_data_list)
        model = self.train_igel_model(G)

        # Detach the embeddings so that we don't backprop at all -- harder baseline!
        G_inference = self.global_graph(data_list)
        embeddings = torch.Tensor(model(G_inference.vs, G_inference).cpu().detach().numpy())

        # Add node features where required
        for field in self.node_feature_fields:
            base_features = getattr(data_list, field, None)
            if base_features is None:
                continue
            concat_result = torch.cat([base_features, embeddings], axis=-1)
            setattr(data_list, field, concat_result)

        # Add edge features where required
        if self.edge_feature_map:
            for index_field, feature_field in self.edge_feature_map.items():
                edge_indices = getattr(data_list, index_field, None)
                edge_attr = getattr(data_list, feature_field, None)
                if edge_indices is None or edge_attr is None:
                    continue

                edge_pair_emb = embeddings[edge_indices]
                edge_emb = self.edge_fn(edge_pair_emb[0], edge_pair_emb[1])
                concat_result = torch.cat([edge_attr, edge_emb], axis=-1)
                setattr(data_list, feature_field, concat_result)
        return data_list

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
        # Note: Imported from IGEL.gnnml-comparisons
        from igel_embedder import get_unsupervised_embedder, TRAINING_OPTIONS
        import torch.nn as nn

        # Do not train if we are using the encoding
        embedder_length = self.vector_length
        if self.use_encoding:
            TRAINING_OPTIONS.epochs = 0
            embedder_length = 0

        # Get the model using the modified training options (with/without epochs)
        trained_model = get_unsupervised_embedder(G, 
                                         self.distance, 
                                         self.seed,
                                         embedder_length,
                                         self.use_relative_degrees,
                                         train_opts=TRAINING_OPTIONS)

        # If we are using encodings, override the parameter matrix
        if self.use_encoding:
            encoder = trained_model.structural_mapper
            embedder_length = encoder.num_elements()

            # Using encodings: simply replace the learned embedding matrix with an identity matrix
            # This will return the structural features as-is.
            input_identity = torch.eye(embedder_length)
            trained_model.matrix = nn.Parameter(input_identity, requires_grad=False).to(trained_model.device)
            trained_model.output_size = embedder_length
        print(f'Prepared IGEL model (encoding-only: {self.use_encoding}) with {embedder_length} features.')
        return trained_model


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
