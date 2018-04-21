import scipy.io as sio
import numpy as np  ##linear algebra
import os           ## manipulate files and directories
import dicom        ## communicating medical images and related information


def load_dcm_profiles(folder_path):
    """
    Reads all profiles with .dcm extension off the disk.
    
    Cao
    """
    # obtain the list of folders.
    folder_list = os.listdir(folder_path)
    folder_list = np.sort(folder_list)
    all_profiles=[]
    for k in range(0, np.size(folder_list)):
        # Return a list, containing the names of 
        # the entries in the directory given by path
        if os.path.isdir(os.path.join(folder_path, folder_list[k])):
            file_path = os.path.join(folder_path, folder_list[k])
            # obtain list of folder names in a folder.
            list_file = os.listdir(file_path)
            for file_name in list_file:
                if file_name.endswith('.dcm'):
                    patient_profile = dicom.read_file(os.path.join(file_path, file_name))
                    if patient_profile is not None:
                        all_profiles.append(patient_profile)
    return all_profiles


def loadNamedMatrix(filename, name):
    """
    Read a MATLAB-formatted matrix from a file, and extract the
    given named variable and return its value.
    """
    try:
        mat = sio.loadmat(filename)
        m = mat[name]
    except:
        print("ERROR: could not load matrix "+filename)
        m = None
    return m


def saveMatrix(filename, matDict):
    """
    Write a MATLAB-formatted matrix file given a dictionary of
    variables.
    """
    try:
        sio.savemat(filename, matDict)
    except:
        print("ERROR: could not write matrix file "+filename)




