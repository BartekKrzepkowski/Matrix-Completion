import torch

def get_matrix_parametrization(model):
    weight = None
    for fc in model.children():
        assert isinstance(fc, torch.nn.Linear) and fc.bias is None
        if weight is None:
            weight = fc.weight.t()
        else:
            weight = fc(weight)
    return weight 