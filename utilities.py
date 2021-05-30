from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import PIL


def axisEqual3D(ax):
    # A hack to make 3D aspect ratio equal in all axis
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_3d(data, simple, close):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xline = data[:, 0]
    yline = data[:, 1]
    zline = data[:, 2]

    if close:
        index = np.where(data[:-1,:] == data[1:,:])[0]
        if len(index) != 0:
            xline = xline[max(0,index[0] - 10000):min(index[0],len(xline))]
            yline = yline[max(0,index[0] - 10000):min(index[0],len(yline))]
            zline = zline[max(0,index[0] - 10000):min(index[0],len(zline))]

    # Create a sphere
    bm = PIL.Image.open('Images/bluemarble.jpg')
    bm = np.array(bm.resize([int(d / 16) for d in bm.size])) / 256

    r = 2
    pi = np.pi
    # coordinates of the image - don't know if this is entirely accurate, but probably close
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
    x = r* np.outer(np.cos(lons), np.cos(lats)).T
    y = r*np.outer(np.sin(lons), np.cos(lats)).T
    z = r*np.outer(np.ones(np.size(lons)), np.sin(lats)).T

    ax.plot3D(xline, yline, zline, 'red')

    if simple:
        ax.plot_surface(
            x, y, z, rstride=1, cstride=1, color='b', alpha=0.75, linewidth=0)
    else:
        ax.plot_surface(
            x, y, z, rstride=1, cstride=1, facecolors=bm, alpha=0.75, linewidth=0)

    if close:
        # Only plot close to Earth
        ax.axes.set_xlim3d(left=-6, right=6)
        ax.axes.set_ylim3d(bottom=-6, top=6)
        ax.axes.set_zlim3d(bottom=-6, top=6)
    else:
        # Plot completely
        axisEqual3D(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

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


