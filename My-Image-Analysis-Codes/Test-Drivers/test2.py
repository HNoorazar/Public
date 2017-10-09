import numpy as np


"""
def extract_sub_center(image_matrices, center_coor, margin_size):
    row_start = center_coor[0] - 1 - margin_size
    row_end   = center_coor[0] + margin_size
    col_start = center_coor[1] - 1 - margin_size
    col_end   = center_coor[1] + margin_size
    return image_matrices[:, row_start:row_end, col_start:col_end ]

A = np.arange(1,101).reshape(10,10)
B = np.zeros((4,10,10))
B[:,:,:] = A
print extract_sub_center(B, [5,5], 2)
"""
