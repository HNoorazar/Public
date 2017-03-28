import numpy as np

def weights_no_coupling(pop_size, ntopics):
    """
    Weights with no coupling: zeros for all off diagonal elements of the
    matrix for each individual.
    """
    return np.tile(np.eye(ntopics, ntopics), (pop_size, 1, 1))

def weights_coupling_same(pop_size, ntopics):
    """
    Weights with random couplings, where every individual shares the
    same coupling matrix.
    """
    weights = (2 * np.random.rand(ntopics, ntopics)) - 1
    signw = np.sign(weights)
    weights = np.multiply(weights, np.transpose(signw))
    np.fill_diagonal(weights, 1)
    weights = np.tile(weights, (pop_size, 1, 1))
    return weights

def weights_coupling(pop_size, ntopics):
    """
    Weights with random couplings, where every individual has its own
    distinct coupling matrix.
    """
    weights = (2 * np.random.rand(pop_size, ntopics, ntopics)) - 1
    for ii in range(pop_size):
        signw = np.sign(weights[ii, :, :])
        weights[ii, :, :] = np.multiply(weights[ii, :, :], np.transpose(signw))
        np.fill_diagonal(weights[ii, :, :], 1)
    return weights
    