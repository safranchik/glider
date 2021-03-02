import torch
import torch.nn as nn


class TwoLayerNN(nn.Module):

    def __init__(self, in_features, hidden_size, out_classes, activation='ReLU'):
        super(TwoLayerNN, self).__init__()

        self.activation = getattr(torch.nn, activation)()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_classes)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)

