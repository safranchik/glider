import torch
import torch.nn as nn


class LogisticRegression(nn.Module):

    def __init__(self, in_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class EBLogisticRegression(nn.Module):

    def __init__(self, num_classes, embed_dim, _data_module):

        super(EBLogisticRegression, self).__init__()

        vocab_size = len(_data_module.vocab)

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_classes)

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x, offsets):
        embedded = self.embedding(x, offsets)
        return self.fc(embedded)

