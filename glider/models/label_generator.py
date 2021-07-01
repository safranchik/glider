import torch.nn as nn
import torch
import torch.nn.functional as F

class LabelCorrectionNetwork(nn.Module):

    def __init__(self, num_labels: int, *args):
        super(LabelCorrectionNetwork, self).__init__()

        """ Standard hidden dimensions are 768 for text datasets and 64 otherise"""

        self.num_labels = num_labels
        self.one_hot = one_hot

        if one_hot:
            self.embedding = self.embedding = nn.Linear(num_labels, embed_dim)
        else:
            self.embedding = nn.Embedding(num_labels, embed_dim)

        self.net = nn.Sequential(
            nn.Linear(embed_dim + x_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, num_labels)
        )

    def forward(self, features, labels):

        # ensures consistency when using the encoder
        if labels.dim() == 1 and self.one_hot:
            labels = F.one_hot(labels, num_classes=self.num_labels)

        x = torch.hstack([features, self.embedding(labels.float())])
        return self.net(x)


