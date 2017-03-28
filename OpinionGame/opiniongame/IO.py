import scipy.io as sio

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
