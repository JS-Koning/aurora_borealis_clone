from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def plot_3d(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xline = data[:, 0]
    yline = data[:, 1]
    zline = data[:, 2]

    ax.plot3D(xline, yline, zline, 'gray')
    plt.show(block=False)