import torch
import torch.nn as nn
from typing import Union, Iterable


class LogisticRegression(nn.Module):

    def __init__(self, in_features: int, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):

    def __init__(self, in_features: int, num_classes: Union[int, Iterable[int]],
                 hidden_layer_sizes: Union[int, Iterable[int]] = None, dropout: float = 0,
                 hidden_activation: str = 'ReLU', output_activation=None):

        super(MLP, self).__init__()

        self.in_features = in_features

        self.drop = None
        if dropout:
            self.drop = nn.Dropout(p=dropout)

        num_classes = [num_classes] if isinstance(num_classes, int) else num_classes

        last_size = in_features
        self.fc_list = nn.ModuleList()
        # generates list of fully connected linear layers if hidden sizes were specified
        if hidden_layer_sizes is not None:
            hidden_layers = [hidden_layer_sizes] if isinstance(hidden_layer_sizes, int) else hidden_layer_sizes

            self.hidden_activation = getattr(torch.nn, hidden_activation)()
            for hidden_size in hidden_layers:
                self.fc_list.append(nn.Linear(last_size, hidden_size))
                last_size = hidden_size

        self.output_activation = getattr(torch.nn, output_activation)() if output_activation is not None else None

        # list of fully connected networks
        self.out_fc_heads = nn.ModuleList([nn.Linear(last_size, c) for c in num_classes])

    def forward(self, x, head=0, output_classifier_features=False):

        # flattens the data if it does not align with the input size (e.g. image tensors with channels)
        if x.shape[1:] != self.in_features:
            x = x.flatten(start_dim=1)

        for fc in self.fc_list:
            x = self.hidden_activation(fc(x))
            if self.drop is not None:
                x = self.drop(x)

        if head == -1:
            output = [head(x) for head in self.out_fc_heads]
        else:
            output = self.out_fc_heads[head](x)

        if self.output_activation is not None:
            output = self.output_activation(output)

        if output_classifier_features:
            return output, x

        return output

