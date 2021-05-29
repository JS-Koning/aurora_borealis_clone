from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def axisEqual3D(ax):
    # A hack to make 3D aspect ratio equal in all axis
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def plot_3d(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xline = data[:, 0]
    yline = data[:, 1]
    zline = data[:, 2]

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)

    ax.plot3D(xline, yline, zline, 'gray')

    axisEqual3D(ax)

    plt.show(block=False)


def plot_3d_near_earth(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xline = data[:, 0]
    yline = data[:, 1]
    zline = data[:, 2]

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)

    ax.plot3D(xline, yline, zline, 'gray')

    axisEqual3D(ax)

    plt.show(block=False)


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


def find_nearest(locations, value):
    """
    find the index of the specified value in the location
    """
    #create distance of earth array
    array_r_2 =locations**2
    polar_r = np.sum(array_r_2, axis=1)
    polar = polar_r**0.5
    #find nearest value 
    #array = np.asarray(polar)
    idx = (np.abs(polar - value)).argmin()
    return idx

