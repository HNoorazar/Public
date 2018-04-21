import numpy as np  ##linear algebra
import os     ## manipulate files and directories
import dicom  ## communicating medical images and related information
import matplotlib.pyplot as plt  ## display the image
import cProfile
import timeit
import scipy as sp 
import scipy.ndimage as nd
import scipy.ndimage
import matplotlib.animation as animation
import matplotlib.image as mpimg
from time import sleep
import glob
import math

def extract_sliceLocation_names(all_profile, no_slices=10):
    """
    This function takes the list of 
    all the patient profiles (all layers, all times),
    and number of layers which by default is set to 10.
    and returns the names of the slices.

    Hossein
    """
    # Create a list of zeros with length of no_slices
    sliceLocation_names = [0] * no_slices

    # counter of number of new slice names.
    count_slicelocation_found = 0
    for ii in xrange(len(all_profile)):
        if count_slicelocation_found == 0:
            sliceLocation_names[count_slicelocation_found] = all_profile[ii].SliceLocation
            count_slicelocation_found += 1
        else:
            if not all_profile[ii].SliceLocation in sliceLocation_names:
                sliceLocation_names[count_slicelocation_found] = all_profile[ii].SliceLocation
                count_slicelocation_found += 1
    return sorted(sliceLocation_names, reverse=True)


def matrix_of_all_times(all_profiles, depth_id, no_time_steps, image_dimension):
    """
    input: all_profiles is list of all images, for example here we have
           imges taken at 10 different depths, over 59 time_steps, so, 
           there are in total 590 images.
           
           no_time_steps is the number of times an image is taken.
           image_dimension is the dimension of image (assumed to be square).
           
    output: a 3D-matrix containing the images taken over time of the same layer.
    
    Takes all the data available, 
    goes through all depths and all time steps, and returns a matrix corresponding to
    a given depth (as an input) at all times!
    """
    image_matrix = np.zeros((no_time_steps, image_dimension, image_dimension))
    count_found = 0
    if len(all_profiles)==0:
        raise ValueError("Profiles are empty, you might see a fully white movie!")
    for j in range(0, len(all_profiles)):
        loc = all_profiles[j].SliceLocation
        if loc == depth_id:
            imgg = all_profiles[j].pixel_array
            image_matrix[count_found, :, :] = imgg
            count_found += 1
    return image_matrix

def extract_2D_submatrix_center(image_matrix, center_coor, margin_size=1):
    """
     This function extracts sub-matrices 
     of a 2D matrix using a pixel as a 
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


def extract_3D_submatrix_center(image_matrix, center_coor, margin_size=1):
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
    input : matrix_2D is the matrix of an image.
            sub_matrix_dim is dimension of the tiles or submatrices we want.
            for example if the original matrix is of size 512 x 512.
            and we want to produce a smaller matrix
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


def find_evolution_of_all_slices(all_profile_list, sliceLocation_names, no_time_steps=59, image_dimensions=[512, 512]):
    """
    inputs:
     1- list of all profiles containing all layers and all times
     2- sliceLocation_names which are names of slices, like -145, -150, etc.
     3- no_time_steps: in our project there are 59 of them.
     4- dimension of images: in our project 512-by-512
    
    output:
     a dictionary with keys equal to "slice" + "sliceName", eg. slice185, slice165, etc.
       (except that slices have negative names like -185, this name is positive)
     whose values are 3D matrices (firsrt two dimensions are image, last dimension is time) 
     corresponding to a given layer at all time shots.
     
     Hossein
    """
    slices_evolutions = {}
    for sliceCount in xrange(len(sliceLocation_names)):
        key_name = "slice" + str(int(np.abs(sliceLocation_names[0])))
        for profile_count in xrange(len(all_profile_list)):
            evolution_matrix = matrix_of_all_times(all_profile_list, 
                                                   sliceLocation_names[sliceCount], 
                                                   no_time_steps, 
                                                   image_dimensions)
            slices_evolutions[key_name] = evolution_matrix
    return slices_evolutions

def rgb2gray(rgb):
    """
    convert an RGB image to B&W using MATLAB's formula!
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.114])

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
"""
############################################################ Cao
"""

##################################
#########  Non_Local_Means
##################################
def non_local_mean(image, constant):
    """
    Yufeng Cao
    """
    [m,n] = np.shape(image)
    Apadded = np.row_stack((np.zeros([1,2*1+n]), np.column_stack((np.zeros([m,1]), 
                            image, np.zeros([m,1]))), np.zeros([1,2*1+n])))
    # obtain neighborhood of each pixel
    neighbordim=[]
    for i in xrange(1,m+1):
        for j in xrange(1,n+1):
            neighbor = Apadded[i-1*1:i+2,j-1*1:j+2]
        neighbordim.append(neighbor)
    new_image=np.reshape(neighbordim,[m*n,9]) # get the big neighborhood matrix
    dist=[]
    for k in xrange(0,m*n):
        for l in xrange(0,m*n):
            d = sum(abs(new_image[k,:] - new_image[l,:]))
            dist.append(d)
    # obtain distint matrix
    new_dist=np.reshape(dist,[m*n,m*n])
    idx=np.argsort(new_dist)
    new_idx=idx[:, 0:constant]  # choose nearest constant col
    new_dist_image=new_image[new_idx]
    # choose the middle point from big neighborhood matrix
    new_mean = np.mean(new_dist_image[:,:,4],1) 
    final_image = np.reshape(new_mean,[m,n])
    return final_image

def region_growing(img, seed):
    """
    The region is iteratively grown by 
    comparing all unallocated neighborhood 
    pixels to the region.

    The difference between a pixel's intensity 
    value and the region's mean, is used as a 
    measure of similarity. 
    The pixel with the smallest difference 
    measured this way is allocated to the 
    respective region. This process stops
    when the intensity difference between 
    region mean and new pixel become larger 
    than a certain treshold.

    The input function is img and seed, where 
    img is the image you wanna segment.
    Note, seed is the interesting point you 
    wanna choose.

    Output is the same size of input, and 
    white part is the area that grow from 
    the seed.
    """
    #Parameters for region growing
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    region_threshold = 20
    region_size = 1
    intensity_difference = 0
    neighbor_points_list = []
    neighbor_intensity_list = []

    #Mean of the segmented region
    region_mean = img[seed]

    #Input image parameters
    height, width = img.shape
    image_size = height * width

    #Initialize segmented output image
    segmented_img = np.zeros([height, width])

    #Region growing until intensity difference becomes greater than certain threshold
    while (intensity_difference < region_threshold) & (region_size < image_size):
        #Loop through neighbor pixels
        for i in range(4):
            #Compute the neighbor pixel position
            x_new = seed[0] + neighbors[i][0]
            y_new = seed[1] + neighbors[i][1]

            #Boundary Condition - check if the coordinates are inside the image
            check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)

            #Add neighbor if inside and not already in segmented_img
            if check_inside:
                if segmented_img[x_new, y_new] == 0:
                    neighbor_points_list.append([x_new, y_new])
                    neighbor_intensity_list.append(img[x_new, y_new])
                    segmented_img[x_new, y_new] = 255

        #Add pixel with intensity nearest to the mean to the region
        distance = abs(neighbor_intensity_list-region_mean)
        pixel_distance = min(distance)
        index = np.where(distance == pixel_distance)[0][0]
        segmented_img[seed[0], seed[1]] = 255
        region_size += 1

        #New region mean
        region_mean = (region_mean*region_size + neighbor_intensity_list[index])/(region_size+1)

        #Update the seed value
        seed = neighbor_points_list[index]
        #Remove the value from the neighborhood lists
        neighbor_intensity_list[index] = neighbor_intensity_list[-1]
        neighbor_points_list[index] = neighbor_points_list[-1]
    return segmented_img

##################################
#########  TV Denoising
##################################

def denoise_tv(image, weight=50, eps=2.e-4, n_iter_max=200):
    ndim = image.ndim
    p = np.zeros((image.ndim, ) + image.shape)
    g = np.zeros_like(p)
    d = np.zeros_like(image)
    i = 0
    while i < n_iter_max:
        if i > 0:
            # d will be the (negative) divergence of p
            ## and the use of slice just make image matrix the same size with gradient matrix.
            d = -p.sum(0)
            slices_d = [slice(None), ] * ndim
            slices_p = [slice(None), ] * (ndim + 1)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax+1] = slice(0, -1)
                slices_p[0] = ax
                d[slices_d] += p[slices_p]
                slices_d[ax] = slice(None)
                slices_p[ax+1] = slice(None)
            out = image + d
        else:
            out = image
        E = (out ** 2).sum()

        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        slices_g = [slice(None), ] * (ndim + 1)
        for ax in range(ndim):
            slices_g[ax+1] = slice(0, -1)
            slices_g[0] = ax
            g[slices_g] = np.diff(out, axis=ax)
            slices_g[ax+1] = slice(None)

        norm = np.sqrt((g ** 2).sum(axis=0))[np.newaxis, ...]
        E += weight * norm.sum()
        tau = 1. / (2.*2.)
        norm *= tau / weight
        norm += 1.
        p -= tau * g
        p /= norm
        E /= float(image.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if np.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out
    
"""
############################################################ Hu
"""
###########################################
#########  L1TV-Norm. A: Dr. Hu did it.
#########             B: Who?
#########             A: yeah, Hu.
#########             Silence!
###########################################
def color2gray(image):
	gray = 0.2125*image[:,:,0] + 0.7154*image[:,:,1] + 0.0721*image[:,:,2]
	return gray

def show_gray_image(image):
	plt.imshow(image, cmap = plt.get_cmap('gray'))
	plt.show()


def gradient(image):
	gx, gy = np.gradient(image)
	return gx, gy


def div0( a, b ):
	""" ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
	with np.errstate(divide='ignore', invalid='ignore'):
		c = np.true_divide( a, b )
		c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
	return c


def smoothing(image, SigmaX, SigmaY):
	return sp.ndimage.filters.gaussian_filter(image, [SigmaX, SigmaY], mode='constant')


def l1_norm_for_matrix(Matrix):
	return abs(Matrix).sum(axis=0).max()

def L1TV_cost(image, approximation,Lambda):
	smoothingAppX, smoothingAppY = gradient(approximation)
	smoothingAppCost = np.power(np.power(smoothingAppX,2) + np.power(smoothingAppY,2),0.5)
	smoothingAppCost = sum(sum(smoothingAppCost))
	approximationCost = sum(sum(abs(approximation-image)))
	return smoothingAppCost + Lambda*approximationCost


def L1TV_gradient(image, approximation, Lambda):
	smoothingAppX, smoothingAppY = gradient(approximation)
	smoothingAppCost = np.power(np.power(smoothingAppX,2) + np.power(smoothingAppY,2),0.5)
	smoothingAppX = div0(smoothingAppX, smoothingAppCost)
	smoothingAppY = div0(smoothingAppY, smoothingAppCost)
	smoothingAppXX, smoothingAppXY = gradient(smoothingAppX)
	smoothingAppYX, smoothingAppYY = gradient(smoothingAppY)
	smoothingGradient = smoothingAppXX + smoothingAppYY
	if np.array_equal(approximation, image):
		approximationgGradient = np.zeros(np.shape(approximation))
	else:
		approximationgGradient = approximation - image
		approximationgGradient = np.sign(approximationgGradient)
	return -smoothingGradient + Lambda*approximationgGradient


def someName(inputImage, sigma_x, sigma_y, Lambdas, maxIteration, epsilon):
    """
    Input: inputImage is a gray scale image matrix.
           sigma_x
           sigma_x
           Lambdas
    Output: 
    """
    # Set initial value for L1TV
    uList = []
    # uDict = {}
    for Lambda in Lambdas: 
        u = smoothing(inputImage, sigma_x, sigma_y)
        
        L1TVCost = [0] * (1+maxIteration)
        L1TVCost[0] = L1TV_cost(inputImage, u, Lambda)
        
        Error = [0] * (1+maxIteration)
        Error[0] = sum(sum(abs(u - inputImage)))
        i = 0
        while i<maxIteration:
            i = i+1
            updateGradient = L1TV_gradient(inputImage, u, Lambda)
            uTempt = u - epsilon * updateGradient
            Error[i] = sum(sum(abs(u - uTempt)))
            L1TVCost[i] = L1TV_cost(inputImage, uTempt, Lambda)
            u = uTempt
        # uDict['lambda-' + str(Lambda)] = u
        uList.append(u)
    return uList

###########################################
######### Matrix Neighbor
###########################################
def matrix_neighbors(Matrix, CenterRow, CenterCol, Radius):
	m,n = np.shape(Matrix)
	# create a mask 
	mask = np.zeros([2*Radius+1, 2*Radius+1])
	mask[mask == 0] = np.NaN
	for i in range(-Radius, Radius + 1):
		for j in range(-Radius, Radius + 1):
			if (CenterRow + i >= 0) & (CenterRow + i <= m+1) & (CenterCol + j >= 0) & (CenterCol + j <= n+1):
				mask[i+Radius, j+Radius] = Matrix[i+CenterRow, j+CenterCol]
			else:
				pass
	return mask

###########################################
######### Label Probability
###########################################
def L2(Array1, Array2):
	Diff = Array1 - Array2
	L2 = math.sqrt(sum(np.power(Diff,2)))
	return L2

def L1(Array1, Array2):
	Diff = abs(Array1 - Array2)
	return sum(Diff)

def label_probability(Array1, Array2, Distance = 'L1'):
	# Array1: each column is a pixel
	# Array2: one reference column
	if Distance == 'L1':
		distance = L1(Array1, Array2)
	elif Distance == 'L2':
		distance = L2(Array1, Array2)
	Probability = np.exp(-distance)
	return Probability

def label_criterion(Probability, threshold):
	indexMax = Probability.index(max(Probability))
	indexMax2 = Probability.index(sorted(Probability)[-2])
	if (Probability[indexMax] - Probability[indexMax2]) >= threshold:
		return indexMax
	else:
		return 100

def final_label(testData, Tumor, Healthy, Vessel, threshold):
    Tumor_Probability = label_probability(testData, Tumor)
    Healthy_Probability = label_probability(testData, Healthy)
    Vessel_Probability = label_probability(testData, Vessel)
    Sum_Probability = Tumor_Probability + Healthy_Probability + Vessel_Probability
    Final_Tumor_Probability = Tumor_Probability/Sum_Probability
    Final_Healthy_Probability = Healthy_Probability/Sum_Probability
    Final_Vessel_Probability = Vessel_Probability/Sum_Probability

    Final_Label = [label_criterion([Final_Tumor_Probability[i], Final_Healthy_Probability[i], Final_Vessel_Probability[i]], 0.4) for i in range(len(Final_Tumor_Probability))]
    return Final_Label


###########################################
######### ColorMap Goes Here
###########################################


"""
############################################################ Enrique
"""
############################################################
#########  Distance_Matrices
############################################################



"""
############################################################ Laramie
"""
def sub(A,B):
    return [x1 - x2 for (x1, x2) in zip(A, B)]

def euclidean_distance(A,B):
    return (math.sqrt(np.dot(sub(A,B),sub(A,B))))

from scipy.spatial import distance_matrix
def distanceMatrix(array1, array2, num):
    return distance_matrix(array1, array2, num)



