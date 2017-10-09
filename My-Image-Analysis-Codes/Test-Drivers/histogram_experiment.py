import matplotlib.pyplot as plt
import imageanalysis.core as cr
import imageanalysis.read_data as rd
import imageanalysis.play_movie as pm
import numpy as np
import os


folder_path = "/Users/hn/Documents/GitHub/Image-Analysis-Working-Group/data"
all_images_list = rd.load_dcm_profiles(folder_path)

time_steps = 59
image_dimension = 512
 
# depth has to be chosen by user!
depth = -165
 
matrix = rd.matrix_of_all_times(all_images_list, depth, time_steps, image_dimension )
one_matrix = matrix[0,:,:]

# concatenate and hstack works the same way.
vector = np.concatenate(one_matrix)
plt.hist(vector, bins=100)  # arguments are passed to np.histogram
plt.title("Histogram with '100' bins")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


matrix = (one_matrix > 0 )*one_matrix
vector = np.concatenate(matrix)
# histV = np.histogram(vector)
plt.hist(vector, bins=100)
plt.show()
# matrixList = [item for subset in matrixList for item in subset]