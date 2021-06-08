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
import h5py


# +
#file_str = 'Datasets/Data_t' + str(time) + 'dt' + str(dt) + \
#                               'n' + str(particles_y * particles_z) + 'relevant'

#file_str = 'testingdataset'
def save_relevant_data(savestring, cutoff_high, cutoff_low):
    """
    Find the indices of the nearest values of cutoff_high, and cutoff_low
    disregards data that is not of interest, saves the stripped data to a file

    Parameters
    ----------
    cutoff_high: float
        finds the nearest index of cutoff value
    cutoff_low: float
        finds the nearest index of cutoff value
    Returns
    -------
    dataset: h5
        Dataset of only the relevant trajectories.



    """

    save_string = 'Datasets/stripped_' + savestring + str(cutoff_low) + '_' + str(cutoff_high) + '.h5'

    earth_distance = cutoff_high

    save_particles_stripped_r = np.zeros((1,3000, 3))
    save_particles_stripped_v = np.zeros((1,3000, 3))
    save_usefull_indices = np.zeros((1,2))

    for i in range(180):

        filestr = 'Datasets/Data_t0.001dt1e-08n32400y'+ str(i)+'.h5'
        print(filestr)
        particles_r, particles_v = utils.load_datafile(filestr)

        distances = np.linalg.norm(particles_r, axis=2)
        velocities = np.linalg.norm(particles_r, axis=2)

        mins = distances.min(axis=1)
        usefull_indices = np.where(mins < earth_distance)
        data_within_range = distances[usefull_indices[0]]
        #print(data_within_range.shape)
        #print(usefull_indices[0])
        #print(data_within_range)
        print(usefull_indices)

        cutoff_indices = np.zeros((len(usefull_indices[0]), 2))
        for i in range(len(usefull_indices[0])):
            cutoff_indices[i, 0] = (utils.find_nearest_index(data_within_range[i, :], cutoff_high))
            cutoff_indices[i, 1] = (utils.find_nearest_index(data_within_range[i, :], cutoff_low))
            # height_data[i, :] = data_within_range[i, cutoff_indices_high : cutoff_indices_low]
        cutoff_indices = cutoff_indices.astype(int)
        #print(cutoff_indices)
        stripped_particles_r = particles_r[usefull_indices[0], :, :]
        stripped_particles_v = particles_v[usefull_indices[0], :, :]

        save_particles_stripped_r = np.concatenate((save_particles_stripped_r, stripped_particles_r))
        save_particles_stripped_v = np.concatenate((save_particles_stripped_v, stripped_particles_v))
        save_usefull_indices = np.concatenate((save_usefull_indices, cutoff_indices))
        #print(cutoff_indices)
        #print(save_particles_stripped_r)
    save_particles_stripped_r = np.delete(save_particles_stripped_r, [0], axis=0)
    save_particles_stripped_v = np.delete(save_particles_stripped_v, [0], axis=0)
    save_usefull_indices = np.delete(save_usefull_indices, [0], axis=0)


    utils.create_datafile_3(save_string, save_particles_stripped_r, save_particles_stripped_v, save_usefull_indices)
    #print(save_usefull_indices)
    print('shape of the indices array found = '+ str(save_usefull_indices.shape))
    return # save_particles_stripped_r, save_particles_stripped_v, save_usefull_indices
# -

save_relevant_data('testingdataset', 1.1, 1.01)

print(save_particles_stripped_r)

mins = distances.min(axis=1)
usefull_indices = np.where(mins < earth_distance)
data_within_range = distances[usefull_indices[0]]
print(data_within_range.shape)
print(usefull_indices[0])
print(data_within_range)

cutoff_indices = np.zeros((len(usefull_indices[0]), 2))
for i in range(len(usefull_indices[0])):
    cutoff_indices[i, 0] = (utils.find_nearest_index(data_within_range[i, :], cutoff_high))
    cutoff_indices[i, 1] = (utils.find_nearest_index(data_within_range[i, :], cutoff_low))
    # height_data[i, :] = data_within_range[i, cutoff_indices_high : cutoff_indices_low]
cutoff_indices = cutoff_indices.astype(int)
print(cutoff_indices)

stripped_particles_r = particles_r[usefull_indices[0], :, :]
stripped_particles_v = particles_v[usefull_indices[0], :, :]


save_particles_stripped_r = np.zeros((1,3000))
print(save_particles_stripped_r)

save_usefull_indices

b = np.array([[1,1], [2,2], [3,3]])
np.delete(b, [1], axis=0)


