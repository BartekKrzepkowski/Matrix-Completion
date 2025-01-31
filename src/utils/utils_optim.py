import torch

def prepare_optim(model, lr):
    return torch.optim.SGD(model.parameters(), lr=lr)