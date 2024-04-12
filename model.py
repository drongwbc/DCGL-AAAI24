import torch
import utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


class DCGL(nn.Module):
    def __init__(self, num_features, hidden_size, n_clusters, emblem_size):
        super(DCGL, self).__init__()
        # Pseudo-siamese network
        self.conv1 = nn.Linear(num_features, hidden_size, bias=False)
        self.conv2 = nn.Linear(hidden_size, emblem_size, bias=False)

        self.ec1 = nn.Linear(num_features, hidden_size, bias=True)
        self.ec2 = nn.Linear(hidden_size, emblem_size, bias=True)

        self.dc1 = nn.Linear(emblem_size, hidden_size, bias=True)
        self.dc2 = nn.Linear(hidden_size, num_features, bias=True)

        # Siamese graph convolution
        self.gbc1 = nn.Linear(emblem_size, int(emblem_size / 2), bias=False)
        self.gbc2 = nn.Linear(int(emblem_size / 2),
                              int(emblem_size / 4),
                              bias=False)
        self.gbc3 = nn.Linear(int(emblem_size / 4), n_clusters, bias=False)

    def forward(self, X, A, n_clusters, nei, upper_nei):
        # Pseudo-siamese network
        h1 = F.relu(self.conv1(A.mm(X)))
        h1 = F.relu(self.conv2(A.mm(h1)))
        h1 = F.normalize(h1, p=2, dim=1)
        S_L, _ = utils.build_LPG(h1.T, nei)

        h2 = F.relu(self.ec1(X))
        h2 = F.relu(self.ec2(h2))
        h2 = F.normalize(h2, p=2, dim=1)

        km = KMeans(n_clusters=n_clusters, n_init=1,
                    init='k-means++').fit(h2.data.cpu().numpy())

        x_rec = F.relu(self.dc1(h2))
        x_rec = F.relu(self.dc2(x_rec))
        x_rec = F.normalize(x_rec, p=2, dim=1)

        # Siamese graph convolution
        G_Lnor = utils.build_GDG(h1, h2, upper_nei)
        S_Lnor = utils.graph_normalize(S_L)
        G_Lnor = utils.graph_normalize(G_Lnor)

        z1 = F.relu(self.gbc1(S_Lnor.mm(h1)))
        z1 = F.relu(self.gbc2(S_Lnor.mm(z1)))
        z1 = F.softmax(self.gbc3(S_Lnor.mm(z1)))

        z2 = F.relu(self.gbc1(G_Lnor.mm(h1)))
        z2 = F.relu(self.gbc2(G_Lnor.mm(z2)))
        z2 = F.softmax(self.gbc3(G_Lnor.mm(z2)))

        return S_L, h1, h2, x_rec, km, z1, z2
