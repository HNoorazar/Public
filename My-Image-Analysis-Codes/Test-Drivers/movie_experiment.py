import imageanalysis.read_data as rd
#import core.play_movie as play
import imageanalysis.core as cr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from time import sleep
import glob
import os




"""  play movie of Yufeng  """
path_of_data = "/Users/L22/Git_hub/Image-Analysis/Data"
all_profiles = rd.load_dcm_profiles(path_of_data)


image_matrix = rd.matrix_of_all_times(all_profiles, -165, 59, 512)
image_matrix = (image_matrix>0) * image_matrix
for k in range(0, 59):
    plt.imshow(image_matrix[k, :, :],cmap='Greys')
    plt.pause(0.5)
plt.show(block=True)


# Accent is good color
"""
Accent, Accent_r, Blues, Blues_r, 
BrBG, BrBG_r, BuGn, BuGn_r, BuPu, 
BuPu_r, CMRmap, CMRmap_r, Dark2, 
Dark2_r, GnBu, GnBu_r, Greens, 
Greens_r, Greys, Greys_r, OrRd, 
OrRd_r, Oranges, Oranges_r, PRGn, 
PRGn_r, Paired, Paired_r, Pastel1, 
Pastel1_r, Pastel2, Pastel2_r, PiYG, 
PiYG_r, PuBu, PuBuGn, PuBuGn_r, 
PuBu_r, PuOr, PuOr_r, PuRd, 
PuRd_r, Purples, Purples_r, 
RdBu, RdBu_r, RdGy, RdGy_r, 
RdPu, RdPu_r, RdYlBu, RdYlBu_r, 
RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, 
Set2, Set2_r, Set3, Set3_r, Spectral, 
Spectral_r, Vega10, Vega10_r, Vega20, 
Vega20_r, Vega20b, Vega20b_r, Vega20c, 
Vega20c_r, Wistia, Wistia_r, YlGn, YlGnBu, 
YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, 
YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, 
binary, binary_r, bone, bone_r, brg, brg_r, 
bwr, bwr_r, cool, cool_r, coolwarm, 
coolwarm_r, copper, copper_r, cubehelix, 
cubehelix_r, flag, flag_r, gist_earth, 
gist_earth_r, gist_gray, gist_gray_r, 
gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, 
gist_rainbow, gist_rainbow_r, gist_stern, 
gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, 
gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, 
hot, hot_r, hsv, hsv_r, inferno, inferno_r, 
jet, jet_r, magma, magma_r, nipy_spectral, 
nipy_spectral_r, ocean, ocean_r, pink, pink_r, 
plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, 
seismic, seismic_r, spectral, spectral_r, 
spring, spring_r, summer, summer_r, 
tab10, tab10_r, tab20, tab20_r, tab20b, 
tab20b_r, tab20c, tab20c_r, terrain, terrain_r, 
viridis, viridis_r, winter, winter_r
"""
