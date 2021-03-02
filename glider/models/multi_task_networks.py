import torch
import torch.nn as nn


class TwoLayerMultiTaskNetwork(nn.Module):

    def __init__(self, in_features=5000, hidden_size=512, out_classes=2):
        super(TwoLayerMultiTaskNetwork, self).__init__()

        self.hidden_activation = torch.nn.LeakyReLU()

        self.fc_1 = nn.Linear(in_features, hidden_size)
        self.fc_2_auxiliary = nn.Linear(hidden_size, out_classes)
        self.fc_2_main = nn.Linear(hidden_size, out_classes)

    def forward_main(self, X):
        X = self.hidden_activation(self.fc_1(X))
        X = self.fc_2_main(X)
        return X

    def forward_auxiliary(self, X):
        X = self.hidden_activation(self.fc_1(X))
        X = self.fc_2_auxiliary(X)
        return X

    def forward(self, X_main, X_aux):
        return self.forward_main(X_main), self.forward_auxiliary(X_aux)

    def accuracy(self, predictions, Y):
        return torch.mean((predictions.argmax(axis=1) == Y).float())
