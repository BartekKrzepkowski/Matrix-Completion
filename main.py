#!/usr/bin/env python3
import os
import numpy
import torch

import matplotlib.pyplot as plt


from src.data.datasets import create_matrix
from src.modules.architectures.models import prepare_model
from src.utils.utils_model import get_matrix_parametrization
from src.utils.utils_data import get_training_data
from src.utils.utils_loss import calc_train_loss, calc_test_loss
from src.utils.utils_optim import prepare_optim
from src.utils.utils_general import set_seed

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(config):
    set_seed(config.seed)
    matrix1 = create_matrix(config.matrix_dim, config.rank1, symmetric=False)
    matrix2 = create_matrix(config.matrix_dim, config.rank2, symmetric=False)
    idxs, target1 = get_training_data(matrix1.numel(), config.n_train_samples, matrix1)
    _, target2 = get_training_data(matrix2.numel(), config.n_train_samples, matrix2, idxs)
    model = prepare_model(hidden_dims=[matrix2.shape[0]]*(config.model_depth+1)).to(device)
    optim = prepare_optim(model, config.lr)
    
    test_losses_critical = []
    
    for critical_end_step in config.critical_end_steps:
        for step in range(config.n_iter):
            parametrized_matrix = get_matrix_parametrization(model)
            with torch.no_grad():
                test_loss = calc_test_loss(parametrized_matrix, matrix1) if step < critical_end_step else calc_test_loss(parametrized_matrix, matrix2)
                
            loss = calc_train_loss(parametrized_matrix, idxs, target1) if step < critical_end_step else calc_train_loss(parametrized_matrix, idxs, target2)
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        test_losses_critical.append(test_loss)
            
        
    plt.figure(figsize=(3.2, 2.7))
    plt.plot(config.critical_end_steps, test_losses_critical, marker='o', markersize=3)
    plt.xlabel('Deficit Switch Epoch')
    plt.ylabel('Test Loss (MSE)')
    plt.ylim(0, max(test_losses_critical) + 0.1)
    plt.savefig(os.path.join('results', f'critical-period-completion.pdf'), bbox_inches='tight')

    # Saving arguments:
    to_save = {'config': config, 'test_losses_critical': test_losses_critical}
    torch.save(to_save, os.path.join('results', 'info.pth'))
    
    
if __name__ == '__main__':
    config = {
        'seed': 817,
        'rank1': 100,
        'rank2': 50,
        'matrix_dim': 100,
        'n_train_samples': 100,     # jaka jest proporcja odpowiednia do wszystkich elementów?
        'model_depth': 4,
        'lr': 1e-1,
        'n_iter': 100,
        'critical_end_steps': [10, 20, 30],
    }