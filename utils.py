import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import scipy.io as scio
from sklearn.cluster import KMeans


def load_data(name):
    path = 'data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0], ))
    X = data['X']
    X = X.astype(np.float32)
    return torch.from_numpy(X).to(dtype=torch.float), labels


def graph_normalize(A):
    """
    Normalize the adj matrix
    """
    degree = torch.sum(A, dim=1).pow(-0.5)
    return (A * degree).t() * degree


def build_LPG(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).to(X.device)
        weights += torch.eye(size).to(X.device)
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.to(X.device)
    weights = weights.to(X.device)
    return weights, raw_weights


def spectral_clustering(A, k, NCut=False, KMrep=1):
    D = np.diag(A.sum(0))
    lap = D - A
    if NCut:
        d_inv = np.linalg.pinv(D)
        sqrt_d_inv = np.sqrt(d_inv)
        lap = np.matmul(np.matmul(sqrt_d_inv, lap), sqrt_d_inv)
    x, V = np.linalg.eig(lap)
    V = np.real(V)
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x: x[0])
    H = np.vstack([V[:, i] for (v, i) in x[:k]]).T
    km = KMeans(n_clusters=k, n_init=KMrep, init='k-means++').fit(H)
    return km


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def build_GDG(h1, h2, nei, transport_rate=0.2):
    """
    Compute global diffusion graph
    """
    h = (h1 + h2) / 2
    G, _ = build_LPG(h.T, nei)

    # graph diffusion
    G_nor = graph_normalize(G)
    G_PPR = transport_rate * torch.linalg.inv(
        torch.eye(G.shape[0]).to(G.device) -
        (1 - transport_rate) * G_nor)

    return G_PPR


def FL_Loss(clusters, h1, centers, label, h2=[], temperature=0.5):
    """
    Feature-level contrastive learning
    """
    loss = 0
    h1 = F.normalize(h1, p=2, dim=1)
    h2 = F.normalize(h2, p=2, dim=1)
    centers = F.normalize(centers, p=2, dim=1)

    indicator = torch.ones(h1.size(0), clusters).to(h1.device)
    for i in range(clusters):
        indicator[i, label[i]] = 0

    sim_positive = torch.exp(torch.mm(h1, h2.T) / temperature)
    sim_negative = torch.mul(
        torch.exp(torch.mm(h1, centers.T) / temperature), indicator)

    loss = - torch.log(torch.diagonal(sim_positive) /
                       (torch.diagonal(sim_positive) + torch.sum(sim_negative, dim=1)))
    return loss.mean()


def GL_Loss(A, h1, par=2e+3):
    """
    Loss of graph learning
    """
    degree = A.sum(dim=1)
    L = torch.diag(degree) - A

    loss = torch.trace(h1.T.matmul(L).matmul(h1)) + (par/2) * torch.norm(A)**2
    return loss.mean()


def CL_Loss(z1, z2, h1, temperature=0.5):
    """
    cluster-level contrastive learning
    """
    u1 = torch.mm(z1.T, h1)
    u1 = F.normalize(u1, p=2, dim=1)

    u2 = torch.mm(z2.T, h1)
    u2 = F.normalize(u2, p=2, dim=1)

    sim_cross1 = torch.exp(torch.mm(u1, u2.T) / temperature)
    sim_cross2 = torch.exp(torch.mm(u2, u1.T) / temperature)

    sim_same1 = torch.exp(torch.mm(u1, u1.T) / temperature)
    sim_same2 = torch.exp(torch.mm(u2, u2.T) / temperature)

    loss = 0
    loss += -torch.log(torch.diagonal(sim_cross1) / torch.sum(sim_cross1, dim=1)
                       ) + (torch.sum(sim_same1, dim=1) - torch.diagonal(sim_same1))
    loss += -torch.log(torch.diagonal(sim_cross2) / torch.sum(sim_cross2, dim=1)
                       ) + (torch.sum(sim_same2, dim=1) - torch.diagonal(sim_same2))

    return loss.mean()
