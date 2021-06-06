"""Import settings"""
import numpy as np
import simulation as sim
import utilities as utils
import matplotlib.pyplot as plt
import timeit
import concurrent.futures as thread
import psutil
import os.path
from os import path

earth_distance = 1
cutoff_high = 1.2
cutoff_low = 1
filestr = 'Datasets\Data_t0.001dt1e-08n10000y0.h5'
particles_r, particles_v = utils.load_datafile(filestr)
print(particles_r.shape)
distances = np.linalg.norm(particles_r, axis=2)
velocities = np.linalg.norm(particles_r, axis=2)
#print(distances.shape)
print(distances.shape)
print(distances)



mins = distances.min(axis=1)
usefull_indices = np.where(mins < earth_distance)
data_within_range = distances[usefull_indices]
print(data_within_range.shape)

height_data = data_within_range
for i in range(len(usefull_indices)):
    cutoff_indices_low = (utils.find_nearest_index(data_within_range, cutoff_high))
    cutoff_indices_high = (utils.find_nearest_index(data_within_range, cutoff_low))
    height_data[i, :] = data_within_range[i, cutoff_indices_high : cutoff_indices_low]

print(height_data)