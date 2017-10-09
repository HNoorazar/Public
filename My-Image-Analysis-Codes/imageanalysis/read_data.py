import numpy as np  ##linear algebra
import os           ## manipulate files and directories
import dicom        ## communicating medical images and related information
import matplotlib.pyplot as plt  ## display the image

"""
Yufeng
          \*  READ DATA  */
"""
# loads all images of all depth and all time steps
def load_dcm_profiles(folderpath):
    # obtain the list of folders.
    folderlist = os.listdir(folderpath)
    all_profiles=[]
    for k in range(0, np.size(folderlist)):
        # Return a list, containing the names of 
        # the entries in the directory given by path
        filepath = os.path.join(folderpath, folderlist[k])
        listfile = os.listdir(filepath)
        for filename in listfile:
            if filename.endswith(".dcm"):
                patint_profile = dicom.read_file(os.path.join("/",filepath,filename))
                if patint_profile is not None:
                    all_profiles.append(patint_profile)
    return all_profiles

# takes all the data available, goes through all depth
# and all time steps, and returns a matrix corresponding to
# a given depth (as an input) at all times!
def matrix_of_all_times(all_profiles, depth_id, no_time_steps, image_dimension):
    """
    input: all_profiles is list of all images, for example here we have
           iamges taken at 10 different depth, over 59 time_steps, so, 
           there are in total 590 images.
           
           no_time_steps is the number of times an image is taken.
           image_dimension is the dimension of image (assumed to be square).
           
    output: a 3D-matrix containing the images taken over time of the same layer.
    """
    image_matrix = np.zeros((no_time_steps, image_dimension, image_dimension))
    count_found = 0
    for j in range(0, len(all_profiles)):
        loc = all_profiles[j].SliceLocation
        if loc == depth_id:
            imgg = all_profiles[j].pixel_array
            image_matrix[count_found, :, :] = imgg
            count_found += 1
    return image_matrix

