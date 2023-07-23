from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import PIL
import h5py
import simulation as sim
from matplotlib import animation, rc


def axis_equal_3d(ax):
    # A hack to make 3D aspect ratio equal in all axis
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def rotate(x, y, z):
    # y - axis rotation https://nl.wikipedia.org/wiki/Rotatiematrix
    # Earth's axial tilt
    theta = sim.theta

    rotation_matrix = np.array(
        [[np.cos(theta), 0, np.sin(theta)],
         [0, 1, 0],
         [-np.sin(theta), 0, np.cos(theta)]]
    )

    x2 = x * rotation_matrix[0][0] + y * rotation_matrix[0][1] + z * rotation_matrix[0][2]
    y2 = x * rotation_matrix[1][0] + y * rotation_matrix[1][1] + z * rotation_matrix[1][2]
    z2 = x * rotation_matrix[2][0] + y * rotation_matrix[2][1] + z * rotation_matrix[2][2]

    return x2, y2, z2


def plot_earth(simple, resolution):
    """"
    Creates a sphere for plotting purposes, can either be earth-like or simply a sphere. Written in a way such that
    trajectories can be added to the plot later on

    Parameters
    ----------
    simple: boolean
    If false, gives a normal translucent sphere, if True sphere is replaced by a translucent perfect spherical earth
    resolution: int
    Resolution of Earth texture if not simple. Preferably given in powers of 2; (max: 1024)

    Returns
    -------
    fig: pyplot
        saves pyplot figure
    ax: pyplot
        pyplot axis, required to add trajectories
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    resolution = 1024 / resolution

    # Create a sphere
    bm = PIL.Image.open('earth.jpg')
    bm = np.array(bm.resize([int(d / resolution) for d in bm.size])) / 256

    # radius
    r = 1
    # spherical wrap
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
    x = r * np.outer(np.cos(lons), np.cos(lats)).T
    y = r * np.outer(np.sin(lons), np.cos(lats)).T
    z = r * np.outer(np.ones(np.size(lons)), np.sin(lats)).T

    x, y, z = rotate(x, y, z)

    if simple:
        ax.plot_surface(
            x, y, z, rstride=1, cstride=1, color='b', alpha=0.75, linewidth=0)
    else:
        ax.plot_surface(
            x, y, z, rstride=1, cstride=1, facecolors=bm, alpha=0.75, linewidth=0)

    ax.set_xlabel("X [$R_{Earth}$]")
    ax.set_ylabel("Y [$R_{Earth}$]")
    ax.set_zlabel("Z [$R_{Earth}$]")

    plt.show(block=False)

    return fig, ax


def plot_3d(ax, data, close):
    """"
    Creates a plot with 3d line-like data to create trajectories for the particles. adds cutoff parameters to delete
    data in the plot that is not usable in most cases. Can be used in combination with utilities.plot_earth

    Parameters
    ----------
    ax: Pyplot
        Axis data for pyplot figures
    data: np.ndarray
        Data that is to be plotted, makes line plots if this data
    close: Boolean
        Plots only near earth trajectories if True, if False plots full trajectories
    Returns
    -------
    None

    """
    xline = data[:, 0]
    yline = data[:, 1]
    zline = data[:, 2]

    ax.plot3D(xline, yline, zline)

    if close:
        # Only plot close to Earth
        ax.axes.set_xlim3d(left=-6, right=6)
        ax.axes.set_ylim3d(bottom=-6, top=6)
        ax.axes.set_zlim3d(bottom=-6, top=6)
    else:
        # Plot completely
        axis_equal_3d(ax)

    plt.show(block=False)


def plot_3d_animation(fig, ax, data, close):
    # choose a different color for each trajectory
    colors = plt.cm.jet(np.linspace(0, len(data)/8, len(data)) % 1.0)
    # set up trajectory lines
    lines = sum([ax.plot([], [], [], '-', linewidth=0.3, c=c) for c in colors], [])
    # set up points
    pts = sum([ax.plot([], [], [], 'o', markersize=0.45, c=c) for c in colors], [])

    # set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(30, 0)
    # initialization function: plot the background of each frame

    def init():
        for line, pt in zip(lines, pts):
            # trajectory lines
            line.set_data([], [])
            # line.set_3d_properties([])
            # points
            pt.set_data([], [])
            # pt.set_3d_properties([])
        return lines + pts

    frame_count = 300

    # animation function.  This will be called sequentially with the frame number
    def animate(i):
        print(i/frame_count)
        # we'll step two time-steps per frame.  This leads to nice results.
        time_steps = data.shape[1]
        # i = (5 * i) % data.shape[1]
        i = int(15*time_steps/16 + int((i+1) / frame_count * time_steps/16))

        for line, pt, xi in zip(lines, pts, data):
            x, y, z = xi[max(0, int(i-time_steps/40)):i].T
            # trajectory lines
            line.set_data(x, y)
            line.set_3d_properties(z)
            # points
            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])
            # stick lines
            # stick_line.set_data(xx,zz)
            # stick_line.set_3d_properties(yy)
        ax.view_init(30, (i+1)*360/frame_count/10)
        fig.canvas.draw()
        return lines + pts

    if close:
        # Only plot close to Earth
        plot_size = 0.75
        ax.axes.set_xlim3d(left=-1.5*plot_size, right=0.5*plot_size)
        ax.axes.set_ylim3d(bottom=-plot_size, top=plot_size)
        ax.axes.set_zlim3d(bottom=-plot_size, top=plot_size)
    else:
        # Plot completely
        axis_equal_3d(ax)

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frame_count, interval=30, blit=True)
    plt.show(block=False)

    return ani


def save_animation(file_name, ani):
    ani.save(file_name, writer='imagemagick', fps=30)
    return True


def custom_space(start, end, num, scaling):
    base = np.linspace(-1.0, 1.0, num=num)

    sign = np.sign(base)

    base = np.power(abs(base), scaling)

    sign = np.where(sign >= 0, sign*end, -sign*start)

    return base * sign


def initialize_loc_vel(init_velocity, distance_earth, offset_y, offset_z):
    """
    Initalize slow solar wind
    init_velocity = m/s
    particles only moving in x position
    distance_earth is the starting distance in the x_direction, offset is based, and scaled, in yz plane
    """
    if init_velocity < 300000:
        print('init_velocity is too low')
    elif init_velocity > 500000:
        print('init_velocity is too high')
    else:
        velocity = np.array([init_velocity, 0, 0])
        loc = np.array([-distance_earth, offset_y, offset_z])
    return velocity, loc


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plot_time_velocity(velocities, dt):
    vel_abs = np.linalg.norm(velocities, axis=1)
    time_array = np.arange(0, dt*len(vel_abs), dt)
    plt.plot(time_array, vel_abs)
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time [s]')
    plt.show()
    return


def plot_time_distance(distances, dt):
    distance_abs = np.linalg.norm(distances, axis=1)
    time_array = np.arange(0, dt*len(distance_abs), dt)
    plt.plot(time_array, distance_abs)
    plt.ylabel('distance [r_earths]')
    plt.xlabel('Time [s]')
    plt.show(block=False)
    return


def create_datafile(file_name, particles_r, particles_v):
    """
        This function creates datasets

        Parameters
        ----------
        file_name: str
            File name for new dataset file
        particles_r: ndarray
            Position data of particles
        particles_v: ndarray
            Velocity data of particles

        Returns
        -------
        True: boolean
            Success
    """

    hf = h5py.File(file_name, 'w')
    hf.create_dataset('particles_positions', data=particles_r, compression="gzip")
    hf.create_dataset('particles_velocities', data=particles_v, compression="gzip")
    hf.close()


def create_datafile_3(file_name, particles_r, particles_v, indices):
    """
        This function creates datasets

        Parameters
        ----------
        file_name: str
            File name for new dataset file
        particles_r: ndarray
            Position data of particles
        particles_v: ndarray
            Velocity data of particles
        indices: ndarray
            indices to be saved

        Returns
        -------
        True: boolean
            Success
    """

    hf = h5py.File(file_name, 'w')
    hf.create_dataset('particles_positions', data=particles_r, compression="gzip")
    hf.create_dataset('particles_velocities', data=particles_v, compression="gzip")
    hf.create_dataset('indices_in_cutoffs', data=indices, compression="gzip")
    hf.close()


def load_datafile(file_name):
    """
        This function loads datasets

        Parameters
        ----------
        file_name: str
            File name for new dataset file

        Returns
        -------
        particles_r: ndarray
            Position data of particles
        particles_v: ndarray
            Velocity data of particles
    """

    # read data set(s)
    hf = h5py.File(file_name, 'r')

    particles_r = hf.get('particles_positions')
    particles_r = np.array(particles_r)

    particles_v = hf.get('particles_velocities')
    particles_v = np.array(particles_v)

    hf.close()

    # numpy.append(tauarray, specific_heat, axis=None) #to add it to array created in line 2 of this 'block'
    return particles_r, particles_v


def save_relevant_data(cutoff_high, cutoff_low, particles_y, time, dt, particles_total, data_points):
    """
    Find the indices of the nearest values of cutoff_high, and cutoff_low
    disregards data that is not of interest, saves the stripped data to a file

    Parameters
    ----------
    cutoff_high: float
        Finds the nearest index of cutoff value
    cutoff_low: float
        Finds the nearest index of cutoff value
    particles_y: ndarray
        Y-grid coordinates
    time: float/int
        Simulation time
    dt: float
        Simulation time-step size
    particles_total: int
        Total amount of particles
    data_points: int
        Amount of data-points in saved data

    Returns
    -------
    True: boolean
        If saved successfully
    """

    save_string = 'Datasets/DataStripped_t' + str(time) + 'dt' + str(dt) + \
                  'n' + str(particles_total) + ".h5"

    earth_distance = cutoff_high

    save_particles_stripped_r = np.zeros((1, data_points, 3))
    save_particles_stripped_v = np.zeros((1, data_points, 3))
    save_useful_indices = np.zeros((1, 2))

    for i in range(particles_y):
        file_str = 'Datasets/Data_t' + str(time) + 'dt' + str(dt) + \
                               'n' + str(particles_total) + 'y' + str(i) + '.h5'
        particles_r, particles_v = load_datafile(file_str)

        distances = np.linalg.norm(particles_r, axis=2)
        # velocities = np.linalg.norm(particles_v, axis=2)

        minimal_distances = distances.min(axis=1)
        useful_indices = np.where(minimal_distances < earth_distance)
        data_within_range = distances[useful_indices[0]]

        cutoff_indices = np.zeros((len(useful_indices[0]), 2))

        for ii in range(len(useful_indices[0])):
            cutoff_indices[ii, 0] = (find_nearest_index(data_within_range[ii, :], cutoff_high))
            cutoff_indices[ii, 1] = (find_nearest_index(data_within_range[ii, :], cutoff_low))

        cutoff_indices = cutoff_indices.astype(int)

        stripped_particles_r = particles_r[useful_indices[0], :, :]
        stripped_particles_v = particles_v[useful_indices[0], :, :]

        save_particles_stripped_r = np.concatenate((save_particles_stripped_r, stripped_particles_r))
        save_particles_stripped_v = np.concatenate((save_particles_stripped_v, stripped_particles_v))
        save_useful_indices = np.concatenate((save_useful_indices, cutoff_indices))

    save_particles_stripped_r = np.delete(save_particles_stripped_r, [0], axis=0)
    save_particles_stripped_v = np.delete(save_particles_stripped_v, [0], axis=0)
    save_useful_indices = np.delete(save_useful_indices, [0], axis=0)

    create_datafile_3(save_string, save_particles_stripped_r, save_particles_stripped_v, save_useful_indices)

    print('Shape of the indices array found: ' + str(save_useful_indices.shape))

    return True


def load_relevant_data(file_name):
    """
    This function loads datasets

    Parameters
    ----------
    file_name: str
        File name for new dataset file

    Returns
    -------
    particles_r: ndarray
        Position data of particles
    particles_v: ndarray
        Velocity data of particles
    indices:
        cutoff indices of relevant data
    """
    
    # read data set(s)
    hf = h5py.File(file_name, 'r')

    particles_r = hf.get('particles_positions')
    particles_r = np.array(particles_r)

    particles_v = hf.get('particles_velocities')
    particles_v = np.array(particles_v)
    
    indices = hf.get('indices_in_cutoffs')
    indices = np.array(indices)

    hf.close()

    return particles_r, particles_v, indices


def probability_absorption():
    energies = np.array([0.4, 0.5, 0.55, 1.0, 1.65, 5.6, 40, 300])
    heights = np.array([270, 250, 210, 170, 150, 120, 100, 75])
    plt.plot(energies, heights)
    plt.show()


def lognormal_dist(sigma, mu, start, stop):
    x = np.linspace(start, stop, 1000)
    pdf = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))
    plt.plot(x, pdf)
    plt.show()
    return


def gasses_absorption(energies):
    """
    Gasses data only works for 10 km height each time.
    returns the height that each particle SHOULD be absorbed at a minimum.
    Data is taken at a geomagnetic pole summer soltice, nighttime. According to the current simulation orientation.

    Input:
    height: int
    returns:

    """
    file_data = np.genfromtxt('atomicoxygen_nitrogen.txt')  # https://ccmc.gsfc.nasa.gov/modelweb/models/msis_vitmo.php
    height = file_data[:, 0]
    ox = file_data[:, 1]
    n2 = file_data[:, 2]

    cutoff_array = np.array([0.4, 0.5, 0.65, 0.1, 1.65, 5.6, 40, 300])  # 1-s2.0-0032063363902526-main%20(2).pdf
    cutoff_height = np.array([210, 190, 170, 150, 130, 110, 90, 70])

    part_cutoffindx = np.zeros(len(energies[:, 0]))
    for i in range(len(energies[:, 0])):
        if np.where(cutoff_array < np.max(energies[i, :]))[0].size == 0:
            part_cutoffindx[i] = 0
        else:
            part_cutoffindx[i] = np.max(np.where(cutoff_array < np.max(energies[i, :])))

    length = len(energies[:, 0])

    index_nasa_cutoff = cutoff_height[part_cutoffindx.astype(int)]/10 - 1
    index_nasa_cutoff = index_nasa_cutoff.astype(int)

    final_index_height = np.zeros(length)
    for i in range(len(part_cutoffindx)):
        particles_num = n2[index_nasa_cutoff[i]:] + ox[index_nasa_cutoff[i]:]
        part_cum = np.cumsum(particles_num[::-1] / sum(particles_num))
        rng = np.random.rand(1)

        final_index_height[i] = np.max(np.where(part_cum < rng))

    heights_final = height[len(height) - final_index_height.astype(int)]

    return heights_final


def location_absorption(part_r, height_locs, indices):
    # Initalization and some basic computations
    distances = np.linalg.norm(part_r, axis=2)
    indices = indices.astype(int)
    r_earth_func = sim.r_earth/1000
    height_locs = height_locs/r_earth_func + 1
    counter1 = 0
    counter2 = 0
    xyz_absorb = np.zeros((len(distances[:, 0]), 3))

    for i in range(len(distances[:, 0])):

        if np.where(distances[i, indices[i, 0]-1: indices[i, 1]] < height_locs[i])[0].size == 0:
            # find average point
            index_average = int(np.round((indices[i,0] + indices[i,1]) / 2.0))
            xyz_absorb[i, :] = part_r[i, index_average]

            # error margin
            delta = 0.001
            passed = False

            while not passed:
                if np.linalg.norm(xyz_absorb[i], axis=0) < 1.01 - delta:
                    print("Error 1", np.linalg.norm(xyz_absorb[i], axis=0))
                    # find the highest point
                    index_average -= 1
                    xyz_absorb[i, :] = part_r[i, index_average]
                    passed = False
                elif np.linalg.norm(xyz_absorb[i], axis=0) > 1.10 + delta:
                    print("Error 2", np.linalg.norm(xyz_absorb[i], axis=0))
                    # find the lowest point
                    index_average += 1
                    xyz_absorb[i, :] = part_r[i, index_average]
                    passed = False
                else:
                    passed = True

            counter1 += 1
        else:
            # Particles that are going to be absorbed
            indice_overall_partial = np.min(np.where(distances[i, indices[i, 0]-1 : indices[i, 1]] < height_locs[i])[0])
            indice_overall = indice_overall_partial + indices[i, 0]-1

            distances_interpolation = np.linspace(distances[i, indice_overall], distances[i, indice_overall+1], num=100)

            x_interpolation = np.linspace(part_r[i, indice_overall-1, 0], part_r[i, indice_overall+1, 0], num=100 )
            y_interpolation = np.linspace(part_r[i, indice_overall-1, 1], part_r[i, indice_overall+1, 1], num=100 )
            z_interpolation = np.linspace(part_r[i, indice_overall-1, 2], part_r[i, indice_overall+1, 2], num=100 )

            index_interpolation = find_nearest_index(distances_interpolation, height_locs[i])

            xyz_absorb[i, 0] = x_interpolation[index_interpolation]
            xyz_absorb[i, 1] = y_interpolation[index_interpolation]
            xyz_absorb[i, 2] = z_interpolation[index_interpolation]

            if np.linalg.norm(xyz_absorb[i], axis=0) < 1.0:
                print("Error 3", np.linalg.norm(xyz_absorb[i], axis=0))

            counter2 += 1

    print('Number of particles by definition not hitting maximum absorption height', counter1)
    print('Number of particles breaching below maximum absorption height', counter2)

    return xyz_absorb


def post_process(part_r, part_v, xyz):
    part_r_new = np.copy(part_r)
    part_v_new = np.copy(part_v)
    for i in range(len(part_r_new)):
        aurora_height = np.linalg.norm(xyz[i], axis=0)
        for ii in range(len(part_r_new[i, :, :])):
            if aurora_height > np.linalg.norm(part_r_new[i, ii, :], axis=0):
                # Alter heights
                part_r_new[i, ii, :] = xyz[i]
                # Alter velocities
                part_v_new[i, ii, :] = np.array([0.0, 0.0, 0.0])

    return part_r_new, part_v_new

def create_plot_errorbar(data, title, x_label, y_label, y_data=None, legend=None, error_bar=None):
    """
        This function creates plots

        Parameters
        ----------
        data: ndarray
            X-data
        title: str
            Text to put above the image
        x_label: str
            X-axis label
        y_label: str
            Y-axis label
        y_data: ndarray
            Y-data
        legend: [str]
            Legends for multiple curves
        error_bar: ndarray
            Error-data

        Returns
        -------
        True: boolean
            Success
    """

    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_data is not None and error_bar is None:
        plt.plot(data, y_data, 'o')
    elif y_data is not None and error_bar is not None:
        plt.errorbar(data, y_data, yerr=error_bar, fmt='o', capsize=3.5, ecolor='red', elinewidth=1.2, ms=2.5,
                     mec='black')
    else:
        plt.plot(data)
    if legend is not None:
        plt.legend(legend)
    plt.show(block=False)
    return True

def angular_incoming(part_r, height):
    distances = np.linalg.norm(part_r, axis=2)
    indices_height = find_nearest_index(distances, height)
    theta = np.arccos(part_r[:, :, 2]/distances[indices_height])
    plt.scatter(theta)