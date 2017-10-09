import imageanalysis.core as cr
import imageanalysis.read_data as rd
import imageanalysis.play_movie as pm
import numpy as np
import os

# get the path of all folders
folder_path = "/Users/hn/Documents/GitHub/Image-Analysis-Working-Group/data"

# obtain the information of all images (590)
all_images_list = rd.load_images(folder_path)
print "np.size(all_images_list) from driver", np.size(all_images_list)

# count the images
num_of_images = len(all_images_list)  


time_steps = 59
image_dimension = 512

# depth has to be chosen by user!
depth = -165

matrix = rd.matrix_of_all_times(all_images_list, depth, time_steps, image_dimension )

print "matrix shape = ", np.shape(matrix)
print "matrix type is", type(matrix)
print matrix[10,10,10]