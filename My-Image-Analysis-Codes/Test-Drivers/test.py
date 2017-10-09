import numpy as np
from math import log, sqrt

def reduce_non_loc_mean(matrix, sub_matrix_dim):
    h, w = matrix.shape
    n = matrix.size / (sub_matrix_dim ** 2)
    block_matrix = (matrix.reshape(h//sub_matrix_dim, sub_matrix_dim, -1, sub_matrix_dim)
               .swapaxes(1,2)
               .reshape(-1, sub_matrix_dim, sub_matrix_dim))
    reduced_matrix = np.mean(np.mean(block_matrix, axis = 1), axis = 1)
    reduced_matrix = reduced_matrix.reshape(int(sqrt(n)), int(sqrt(n)))
    return reduced_matrix

A = np.arange(36).reshape(6,6)
B = reduce_non_loc_mean(A, 2)
print B

print np.shape(B)
print type(B)

"""
# original KEEP this HERE
def blockshaped(arr, nrows, ncols):
    ""
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    ""
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
"""