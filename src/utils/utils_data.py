import numpy as np
import torch


def get_training_data(n, n_train_samples, w_gt, idxs=None):
    if idxs is None :
        indices = torch.multinomial(torch.ones(n * n), n_train_samples, replacement=False)
        us, vs = indices // n, indices % n
        idxs = torch.stack([us, vs], dim=1)
    ys_ = w_gt[idxs[:,0], idxs[:,1]]
    assert 0.8 <= ys_.pow(2).mean().sqrt() <= 1.2   # co to robi?
    return idxs, ys_