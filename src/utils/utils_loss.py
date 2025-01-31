def calc_test_loss(parametrized_matrix, matrix):
    train_loss = (parametrized_matrix - matrix).view(-1).pow(2).mean() / 2
    return train_loss


def calc_train_loss(parametrized_matrix, idxs, target):
    entries = parametrized_matrix[idxs[:, 0], idxs[:, 1]]
    train_loss = (entries - target).pow(2).mean() / 2
    return train_loss