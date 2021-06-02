"""Import settings"""
import numpy as np
import simulation as sim
import utilities as utils
import matplotlib.pyplot as plt
import timeit
import concurrent.futures as thread
import psutil

"""Program settings"""
# Initial trajectory simulation
do_simulation = False
# Particle absorption simulation
do_post_processing = False
# Data processing
do_data_processing = True

"""Multi-threading settings"""
# Enable multi-threading
multi_threading = True
# Automatically decide amount of threads
automatic_threads = False
if automatic_threads:
    # Amount of logical processors the current machine has
    threads = psutil.cpu_count(logical=True)
else:
    threads = 3
# Create multi-threading pool with a maximum of number a threads
executor = thread.ProcessPoolExecutor(threads)

"""Seed settings"""
# Seed for reproducibility ||| Guess that's why they call it 'seed'...
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
save_reduced = True
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
custom_grid = True
position_x = -5E8

particles_y = 150
minimum_y = -1E8
maximum_y = 1E8

particles_z = 150
minimum_z = -1E8
maximum_z = 1E8

"""Initial velocity settings"""
# Initial velocities (uniform distribution) [m/s] ||| https://link.springer.com/article/10.1007/s11207-020-01730-z
# minimum velocity: 300 km/s ||| Slow solar wind @ 0.9 ~ 1.1 AU
minimum_v = 3.0e5
# maximum velocity: 750 km/s ||| Fast solar wind @ 0.9 ~ 1.1 AU
maximum_v = 7.5e5


def main():
    # Start program timer
    time_start = timeit.default_timer()

    # Particles trajectory simulation
    if do_simulation:
        # Begin main particles simulation
        print("Simulating...")

        # Define grid coordinates for Y- and Z-coordinates
        if custom_grid:
            y_space = utils.custom_space(minimum_y, maximum_y, particles_y)
            z_space = utils.custom_space(minimum_z, maximum_z, particles_z)
        else:
            y_space = np.linspace(minimum_y, maximum_y, particles_y)
            z_space = np.linspace(minimum_z, maximum_z, particles_z)

        # Define all initial velocities
        v_init_all = np.array([np.random.uniform(minimum_v, maximum_v, particles_y*particles_z)])

        for y in range(len(y_space)):
            # Initialize arrays for saving reduced simulation data
            particles_r = np.zeros((particles_z, save_data_points, 3))
            particles_v = np.zeros((particles_z, save_data_points, 3))

            futures = []

            for z in range(len(z_space)):
                # Show progress of full particles simulation
                print("Simulating particle:", y * particles_y + z + 1, "out of", particles_y * particles_z)

                # Define grid positions
                position_y = y_space[y]
                position_z = z_space[z]

                # Initialize particle position
                r_init = np.array([position_x, position_y, position_z])
                # Initialize particle velocity
                v_init = np.array([np.random.uniform(minimum_v, maximum_v), 0.0, 0.0])

                # Change particles' charge for symmetry
                charge_factor2 = 1
                if position_y >= 0:
                    # Positrons instead of electrons
                    charge_factor2 = -1

                # Simulate particle
                if multi_threading:
                    future = executor.submit(sim.simulate, r_init, v_init, charge_factor*charge_factor2, mass_factor, dt, time_steps, save_reduced, save_data_points, z)
                    futures.append(future)
                else:
                    r_save_data, v_save_data, z2 = sim.simulate(r_init, v_init, charge_factor*charge_factor2, mass_factor, dt, time_steps, save_reduced, save_data_points, z)

                    particles_r[z, :, :] = r_save_data
                    particles_v[z, :, :] = v_save_data

            if multi_threading:
                futures, _ = thread.wait(futures)
                for future in futures:
                    r_save_data, v_save_data, z = future.result()
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
        # Begin post-processing
        print("Post-processing...")

        # TODO
        # Load data
        # Filter useful trajectories (within 3-Earth Radius)
        # Trace back trajectories to stop at absorption altitudes
        # Save differences in data (to get smaller file sizes...)
        pass

    # Data-processing for showing results
    if do_data_processing:
        # Begin data-processing
        print("Data-processing...")

        # Plot 3D Earth
        fig, ax = utils.plot_earth(plot_simple)

        # Iterate over Y-grid
        for y in range(particles_y):
            # Show progress
            if (y+1) % int(particles_y / 10) == 0:
                print("Loading progress:", ("%.2f" % ((y+1) / particles_y * 100)), "%...")

            # Filename of dataset to load
            file_str = 'Datasets/Data_t' + str(time) + 'dt' + str(dt) + \
                       'n' + str(particles_y * particles_z) + 'y' + str(y) + ".h5"

            # Load dataset
            r_dataset, v_dataset = utils.load_datafile(file_str)

            # Plot relevant trajectories
            for r_data in r_dataset:
                # Plot particle trajectory
                if r_data[-1, 0] ** 2 + r_data[-1, 1] ** 2 + r_data[-1, 2] ** 2 < 1.1 ** 2:
                    # Only plot when end-point is closer to than 3 Earth-radia (ignore deflected particles)
                    utils.plot_3d(ax, r_data, plot_near_earth)

    # End program timer
    time_elapsed = timeit.default_timer() - time_start
    print("Runtime:", time_elapsed, "seconds")

    # Block program from terminating
    plt.show()

    return True


# Execute main program
if __name__ == '__main__':
    if main():
        exit()
