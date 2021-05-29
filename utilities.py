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