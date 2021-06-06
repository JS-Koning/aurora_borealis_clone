from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import PIL
import h5py
import simulation as sim

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


def plot_earth(simple):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Create a sphere
    bm = PIL.Image.open('earth.jpg')
    bm = np.array(bm.resize([int(d / 16) for d in bm.size])) / 256

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

