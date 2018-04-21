import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from time import sleep
import glob
import os

def play_movie(image_matrix, time_range=[0, 59], color_map = 'Greys', pause_time=0.01):
    """
    This function takes a 3D matrix and plays it as a movie.
    start_time is the slice we want the movie to start. it has to be` between 0-57.
    end_time is the slice which we want to stop at. it has to be between 1-58.
    
    Yunfeng
    """
    count = 0
    for time in xrange(time_range[0], time_range[1]):
        plt.imshow(image_matrix[time, :, :], cmap= color_map)
        plt.pause(pause_time)
        count += 1
    plt.show(block=True)
