import numpy as np
import torch

def create_matrix(n, rank, symmetric=False):
    r = rank
    U = np.random.randn(n, r).astype(np.float32)
    if symmetric:
        V = U
    else:
        V = np.random.randn(n, r).astype(np.float32)
    w_gt = U.dot(V.T) / np.sqrt(r)
    w_gt = w_gt / np.linalg.norm(w_gt, 'fro') * n
    return torch.from_numpy(w_gt)  # w_gt