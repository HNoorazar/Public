import numpy as np  ##linear algebra
import os     ## manipulate files and directories
import dicom  ## communicating medical images and related information
import matplotlib.pyplot as plt  ## display the image


def extract_2D_submatrix_center(image_matrix, center_coor, margin_size=1):
    """
     This function extracts sub-matrices 
     of a 3D matrix using a pixel as a 
     center, and using margin_size as 
     the number of pixels around the 
     center pixel.
     default margin_size is 1.
     
     Hossein
    """
    row_start = center_coor[0] - margin_size
    row_end   = center_coor[0] + margin_size + 1
    col_start = center_coor[1] - margin_size
    col_end   = center_coor[1] + margin_size + 1
    return image_matrix[row_start:row_end, col_start:col_end ]


def extract_3D_submatrix_center(image_matrix, center_coor, margin_size):
    """
    This function extracts sub-matrices of a 3D matrix.
    Hossein
    """
    row_start = center_coor[0] - margin_size
    row_end   = center_coor[0] + margin_size + 1
    col_start = center_coor[1] - margin_size
    col_end   = center_coor[1] + margin_size + 1
    return image_matrix[:, row_start:row_end, col_start:col_end ]



def extract_2D_submatrix_upper(image_matrices, upper_left, sub_size):
    """
    input: image_matrices : which is 3D matrix, output of matrix_of_all_times.
           upper_left     : coordinate of the upper_left pixel to start cutting
                            it is a list. it Has to be provided like upper_left = [2, 3]
                            so, each entry is of type 'int'.
           sub_size       : size of the submatrix we want!

    REASON: The reason I did upper_left and isize, and not the way Kevin wanted
            is that if we want a 2-by-2 matrix, then there is no center!
    
    output: a submatrix of the matrix image_matrices on the given time interval
    
    Hossein
    """
    start_row = upper_left[0]
    end_row   = start_row + sub_size

    start_col = upper_left[1]
    end_col   = start_col + sub_size
    return image_matrices[start_row:end_row, start_col:end_col]


def extract_3D_submatrix_upper(image_matrices, upper_left, sub_size):
    """
    input: image_matrices : which is 3D matrix, output of matrix_of_all_times.
           upper_left     : coordinate of the upper_left pixel to start cutting
                            it is a list. it Has to be provided like upper_left = [2, 3]
                            so, each entry is of type 'int'.
           sub_size       : size of the submatrix we want!

    REASON: The reason I did upper_left and isize, and not the way Kevin wanted
            is that if we want a 2-by-2 matrix, then there is no center!
    
    output: a submatrix of the matrix image_matrices on the given time interval
    
    Hossein
    """
    start_row = upper_left[0]
    end_row   = start_row + sub_size

    start_col = upper_left[1]
    end_col   = start_col + sub_size

    return image_matrices[:, start_row:end_row, start_col:end_col]


def aggregate_2D(matrix_2D, sub_matrix_dim):
    """
    This funciton takes a 2D square matrix, and returns a 2D square matrix.
    input : matrix is the matrix of an image.
            sub_matrix_dim is dimensiokn of the tiles or submatrices we want.
            for example if the original matrix is of size 512 x 512.
            and we want to produce a smaller matrix by non-local-mean 
            by averaging over entries of submatrices of size 2 x 2 to get 
            a matrix of size 128 x 128
            
    output: a matrix of size m x m where m = np.shape(matrix)[0] / sub_matrix_dim
    
    Hossein
    """
    h, w = matrix_2D.shape
    n = matrix_2D.size / (sub_matrix_dim ** 2)
    block_matrix = (matrix_2D.reshape(h//sub_matrix_dim, sub_matrix_dim, -1, sub_matrix_dim)
                    .swapaxes(1,2)
                    .reshape(-1, sub_matrix_dim, sub_matrix_dim))
    reduced_matrix = np.mean(np.mean(block_matrix, axis = 1), axis = 1)
    reduced_matrix = reduced_matrix.reshape(int(sqrt(n)), int(sqrt(n)))
    return reduced_matrix


def aggregate_3D(matrix_3D, sub_matrix_dim):
    """
    input:   matrix_3D is the matrix of the same depth at different times.
             matrix_3D.shape = time_size, image_dimension. 
             In our example time = 59, and image_dimension = 512.
           
             sub_matrix_dim is the dimension of the submatrices we want to extract.
    output:  the aggregated matrix. 
    
    Hossein
    """
    (time_size, no_rows, no_col) = matrix_3D.shape
    reduced_mat_size = no_rows / sub_matrix_dim
    reduced_matrix = np.zeros((time_size, reduced_mat_size, reduced_mat_size))
    
    for time in xrange(time_size):
        reduced_matrix[time, :, :] = aggregate_2D(matrix_3D[time, :,:], sub_matrix_dim)
    return reduced_matrix


#########
#########  Non_Local_Means
#########
def vectorize_tiles(image_matrix, margin_size=1):
    """
    This function takes a 3D matrix and the 
    size of the margin about the center pixel 
    as an input, and generates a matrix whose 
    columns are entries of the submatrices with 
    a_{ij} as center and the boundaries about it.
    
    ** The centers move along rows. So, sequence of centers
    would be A_{2,2}, A_{2,3}, ..., A_{2,n-1},
             A_{3,2}, A_{3,3}, ..., A_{3,n-1},

    It is assumed that image_matrix is a square matrix!
    
    Hossein
    """
    # make sure image_matrix is a numpy array. (Yufeng tends to create lists! :D)
    image_matrix = np.asarray(image_matrix)
    (no_row, no_col) = image_matrix.shape
    no_submatrices = (no_row - (2 * margin_size)) ** 2
    sub_matrix_size = 1 + (2 * margin_size)
    vectorized_subs = np.zeros((sub_matrix_size**2, no_submatrices))
    count = 0
    for row_count in xrange(margin_size, no_row - margin_size):
        for col_count in xrange(margin_size, no_row - margin_size):
            vectorized_subs[:,count] = np.concatenate(extract_2Dsubmatrix_center(image_matrix, 
                                                         [row_count, col_count], 
                                                         margin_size))
            count += 1
    return vectorized_subs












#########
#########  L1TV-Norm
#########










#########
#########  Distance_Matrices
#########
