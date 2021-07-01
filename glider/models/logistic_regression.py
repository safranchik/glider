import torch.nn as nn

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

