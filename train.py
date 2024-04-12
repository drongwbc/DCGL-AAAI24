import torch
import utils
import torch.nn.functional as F
import numpy as np
import random
import torch.nn as nn
import warnings
import math
import time
from model import DCGL
from torch.optim import Adam
from evaluation import eva


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args):
    setup_seed(args.seed)
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Data process
    [X, Y] = utils.load_data(args.name)
    args.samples = X.shape[0]
    args.input_dim = X.shape[1]
    args.n_clusters = len(np.unique(Y))
    args.upperNei = math.floor(args.samples / args.n_clusters)
    X = F.normalize(X, p=2, dim=1).to(device)

    nei = args.neighbor
    A, _ = utils.build_LPG(X.T, nei)
    A = utils.graph_normalize(A).to(device)

    curModel = DCGL(num_features=args.input_dim,
                    hidden_size=args.hidden_size,
                    emblem_size=args.emblem_size,
                    n_clusters=args.n_clusters).to(device)
    print(curModel)
    print(
        f'Samples: {X.shape[0]}, Dimensions: {args.input_dim}, Clusters: {args.n_clusters}'
    )
    print("---------------------")

    optimizer = Adam(curModel.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)

    MSE = nn.MSELoss()
    retScore = [0, 0, 0, 0]
    # Start train
    start_time = time.time()
    for epoch in range(args.max_epoch):
        curModel.train()
        S_L, h1, h2, X_rec, km, z1, z2 = curModel(X, A, args.n_clusters, nei,
                                                  args.upperNei)

        l1 = MSE(X, X_rec)

        label = torch.LongTensor(km.labels_).to(device)
        centers = torch.FloatTensor(km.cluster_centers_).to(device)
        l2 = utils.FL_Loss(args.n_clusters, h1, centers, label, h2)

        l3 = utils.GL_Loss(S_L, h1)

        l4 = utils.CL_Loss(z1, z2, h1)

        loss = l1 + l2 + args.lambda1 * l3 + args.lambda2 * l4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            S_L, h1, _, _, _, _, _ = curModel(X,
                                              A,
                                              args.n_clusters,
                                              nei=nei,
                                              upper_nei=args.upperNei)

            label = utils.spectral_clustering(S_L.cpu().numpy(),
                                              args.n_clusters,
                                              NCut=True).labels_
            acc, nmi, ari, f1 = eva(Y, label)
            print(
                f'epoch {epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}'
            )
            retScore[0] = max(retScore[0], acc)
            retScore[1] = max(retScore[1], nmi)
            retScore[2] = max(retScore[2], ari)
            retScore[3] = max(retScore[3], f1)

            if (epoch + 1) % args.update_interval == 0:
                if nei >= args.upperNei:
                    break
                else:
                    nei += args.neighbor
            
    end_time = time.time()
    running_time = end_time - start_time

    print("---------------------")
    print(
        f'final acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, f1: {f1:.4f}')
    print(
        f'best  acc: {retScore[0]:.4f}, nmi: {retScore[1]:.4f}, ari: {retScore[2]:.4f}, f1: {retScore[3]:.4f}'
    )
    print(f'Running time: {running_time} 秒')
