"""Import settings"""
import numpy as np
import simulation as sim
import utilities as utils
import matplotlib.pyplot as plt
import timeit

"""Program settings"""
# Initial trajectory simulation
do_simulation = False
# Particle absorption simulation
do_post_processing = True
# Data processing
do_data_processing = True

"""Seed settings"""
# Seed for reproducibility
seed = 420
np.random.seed(seed)

"""Time settings"""
# Time between each iteration ||| minimum: 1E-6 ||| okay: 1E-7 ||| good: 1E-8 ||| best: 1E-9
dt = 1E-8
# Total time length [s]
time = 1E-3
# Time steps
time_steps = int(time/dt)

"""Dataset settings"""
# Dataset settings
save_data = True
save_data_points = round(3 * 1E-5/dt)

"""Plot settings"""
# Plot settings
plot_simple = False
plot_near_earth = True

"""Particle settings"""
# Factors to initialize (relativistic) charged particle with
relativistic = False
mass_factor = 1.0
charge_factor = 1.0

"""Initial position settings"""
# Initial positions grid [m] ||| Approximately 0.0033 AU (=5E8 m) from Earth center
position_x = -5E8

particles_y = 100
minimum_y = -1E8
maximum_y = 1E8

particles_z = 100
minimum_z = -1E8
maximum_z = 1E8

"""Initial velocity settings"""
# Initial velocities (uniform distribution) [m/s] ||| https://link.springer.com/article/10.1007/s11207-020-01730-z
# minimum velocity: 300 km/s [slow solar wind @ 0.9~1.1 AU]
minimum_v = 3.0e5
# maximum velocity: 750 km/s [fast solar wind @ 0.9~1.1 AU]
maximum_v = 7.5e5


def main():
    # Start program timer
    time_start = timeit.default_timer()

    # Particles trajectory simulation
    if do_simulation:
        y_space = np.linspace(minimum_y,maximum_y,particles_y)
        z_space = np.linspace(minimum_z,maximum_z,particles_z)

        for y in range(len(y_space)):
            particles_r = np.zeros((particles_z, save_data_points, 3))
            particles_v = np.zeros((particles_z, save_data_points, 3))

            for z in range(len(z_space)):
                print("Simulating particle:", y*particles_y + z + 1, "out of", particles_y*particles_z)

                # Define grid positions
                position_y = y_space[y]
                position_z = z_space[z]

                # Initialize particle position
                r_init = np.array([position_x, position_y, position_z])
                # Initialize particle velocity
                v_init = np.array([np.random.uniform(minimum_v, maximum_v)])

                # Change particles' charge for symmetry
                charge_factor2 = 1
                if position_y >= 0:
                    # Positrons instead of electrons
                    charge_factor2 = -1

                # Simulate particle
                r_data, v_data = sim.simulate(r_init, v_init, charge_factor*charge_factor2, mass_factor, dt, time_steps)

                index = np.where(r_data[:-1, :] == r_data[1:, :])[0]

                if len(index) != 0:
                    r_save_data = r_data[max(0, index[3] - save_data_points):min(index[3], len(r_data) - 1), :]
                    v_save_data = v_data[max(0, index[3] - save_data_points):min(index[3], len(v_data) - 1), :]
                else:
                    r_save_data = r_data[len(r_data) - 1 - save_data_points:len(r_data) - 1, :]
                    v_save_data = v_data[len(v_data) - 1 - save_data_points:len(v_data) - 1, :]

                particles_r[z, :, :] = r_save_data
                particles_v[z, :, :] = v_save_data

            # Save simulation data for every y (not all at once to avoid memory issues)
            if save_data:
                # Save data for current y
                file_str = 'Datasets/Data_t' + str(time) + 'dt' + str(dt) + \
                           'n' + str(particles_y * particles_z) + 'y' + str(y) + ".h5"
                utils.create_datafile(file_str, particles_r, particles_v)

    # Particles absorption processing
    if do_post_processing:
        # Load data
        # Filter useful trajectories (within 3-Earth Radius)
        # Trace back trajectories to stop at absorption altitudes
        # Save differences in data (to get smaller file sizes...)
        pass

    # Data-processing for showing results
    if do_data_processing:
        print("Data-processing...")
        # Plot 3D Earth
        fig, ax = utils.plot_earth(plot_simple)

        for y in range(particles_y):
            if (y+1) % int(particles_y / 10) == 0:
                print("Loading progress:", ("%.2f" % ((y+1) / particles_y * 100)), "%...")

            file_str = 'Datasets/Data_t' + str(time) + 'dt' + str(dt) + \
                       'n' + str(particles_y * particles_z) + 'y' + str(y) + ".h5"
            r_dataset, v_dataset = utils.load_datafile(file_str)

            for r_data in r_dataset:
                # Plot particle trajectory
                if r_data[-1, 0] ** 2 + r_data[-1, 1] ** 2 + r_data[-1, 2] ** 2 < 99 ** 2:
                    # Only plot when end-point is closer to than 3 Earth-radia (ignore deflected particles)
                    utils.plot_3d(ax, r_data, plot_near_earth)

    # End program timer
    time_elapsed = timeit.default_timer() - time_start
    print("Runtime:", time_elapsed, "seconds")

    # Block program from terminating
    plt.show()
    return time_elapsed


main()
