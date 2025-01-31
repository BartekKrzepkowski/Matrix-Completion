import torch

def prepare_model(hidden_dims):
    model = torch.nn.Sequential(*[torch.nn.Linear(f_in, f_out, bias=False) for (f_in, f_out) in hidden_dims])
    return model