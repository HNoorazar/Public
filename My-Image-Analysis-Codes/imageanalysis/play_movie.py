import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from time import sleep
import glob
import os

"""
Yunfeng
"""
def play_movie(image_matrix, start_time, end_time, color = 'Greys' ):
    """
    This function takes a 3D matrix and plays it as a movie.
    start_time is the slice we want the movie to start. it has to be` between 0-57.
    end_time is the slice which we want to stop at. it has to be between 1-58.
    """
    for time in xrange(start_time, end_time+1):
        plt.imshow(image_matrix[time, :, :], cmap= color)
        plt.pause(0.5)
    plt.show(block=True)
