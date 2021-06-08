from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import PIL
import h5py
import simulation as sim
from matplotlib import animation, rc

def axisEqual3D(ax):
    # A hack to make 3D aspect ratio equal in all axis
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
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
    Boolean: True/False
    If false, gives a normal translucent sphere, if True sphere is replaced by a translucent perfect spherical earth

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

    x,y,z = rotate(x,y,z)

    if simple:
        ax.plot_surface(
            x, y, z, rstride=1, cstride=1, color='b', alpha=0.75, linewidth=0)
    else:
        ax.plot_surface(
            x, y, z, rstride=1, cstride=1, facecolors=bm, alpha=0.75, linewidth=0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

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
        axisEqual3D(ax)

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
            #line.set_3d_properties([])
            # points
            pt.set_data([], [])
            #pt.set_3d_properties([])
        return lines + pts

    frame_count = 300

    # animation function.  This will be called sequentially with the frame number
    def animate(i):
        # we'll step two time-steps per frame.  This leads to nice results.
        time_steps = data.shape[1]
        #i = (5 * i) % data.shape[1]
        i = int(7*time_steps/8 + int((i+1) / frame_count * time_steps/8))

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
        ax.view_init(30, (i+1)*360/frame_count)
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
        axisEqual3D(ax)

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
    if init_velocity<300000:
        print('init_velocity is too low')
    elif init_velocity>500000:
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


def save_relevant_data(savestring, cutoff_high, cutoff_low, particles_y):
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
        Dataset of the trajectories with indices closest to cutoff strings.


    """

    save_string = 'Datasets/stripped_' + savestring + str(cutoff_low) + '_' + str(cutoff_high) + '.h5'

    earth_distance = cutoff_high

    save_particles_stripped_r = np.zeros((1,3000, 3))
    save_particles_stripped_v = np.zeros((1,3000, 3))
    save_usefull_indices = np.zeros((1,2))

    for i in range(particles_y):

        filestr = 'Datasets/Data_t0.001dt1e-08n32400y'+ str(i)+'.h5'
        #print(filestr)
        particles_r, particles_v = load_datafile(filestr)

        distances = np.linalg.norm(particles_r, axis=2)
        velocities = np.linalg.norm(particles_r, axis=2)

        mins = distances.min(axis=1)
        usefull_indices = np.where(mins < earth_distance)
        data_within_range = distances[usefull_indices[0]]
        #print(data_within_range.shape)
        #print(usefull_indices[0])
        #print(data_within_range)
        #print(usefull_indices)

        cutoff_indices = np.zeros((len(usefull_indices[0]), 2))
        for i in range(len(usefull_indices[0])):
            cutoff_indices[i, 0] = (find_nearest_index(data_within_range[i, :], cutoff_high))
            cutoff_indices[i, 1] = (find_nearest_index(data_within_range[i, :], cutoff_low))
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


    create_datafile_3(save_string, save_particles_stripped_r, save_particles_stripped_v, save_usefull_indices)
    #print(save_usefull_indices)
    print('shape of the indices array found = '+ str(save_usefull_indices.shape))
    return # save_particles_stripped_r, save_particles_stripped_v, save_usefull_indices
