import torch
import torch.nn as nn


class LogisticRegression(nn.Module):

    def __init__(self, in_features=5000, out_classes=2):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(in_features, out_classes)

    def forward(self, X):
        return self.fc(X)

    def accuracy(self, predictions, Y):
        return torch.mean((predictions.argmax(axis=1) == Y).float())
