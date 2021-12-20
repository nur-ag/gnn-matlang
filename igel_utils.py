import torch

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
