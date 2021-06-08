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

"""Program settings"""
# Initial trajectory simulation
do_simulation = False
# Particle absorption simulation

do_post_processing = True
do_save_stripped_data = False
do_load_stripped_data = True
do_post_processing = False

# Data processing
do_data_processing = True

"""Logging settings"""
print_simulation_progress = True
print_simulation_initialization = False

"""Multi-threading settings"""
# Enable multi-threading
multi_threading = True
# Automatically decide amount of threads
automatic_threads = True
if automatic_threads:
    # Amount of logical processors the current machine has
    threads = psutil.cpu_count(logical=True)
else:
    # Manual amount of threads
    threads = 4
# Create multi-threading pool with a maximum of number a threads
executor = thread.ProcessPoolExecutor(threads)

"""Seed settings"""
# Seed for reproducibility ||| Guess that's why they call it 'seed'...
seed = 420
np.random.seed(seed)

"""Time settings"""
# Time between each iteration ||| minimum: 1E-6 ||| okay: 1E-7 ||| good: 1E-8 ||| best: 1E-9
dt = 1E-8
# Total time length [s] ||| direct only: at least 1E-3 ||| indirect and direct: at least 5E-1
time = 1E-3
# Time steps
time_steps = int(time/dt)

"""Dataset settings"""
# Dataset settings
save_data = False
save_reduced = True
save_data_points = round(3 * 1E-5/dt)

"""Plot settings"""
# Plot settings
plot_simple = False
# Resolution of Earth texture ||| Multiple of 2 up until 1024
plot_earth_resolution = 64
# Resolution of Earth texture ||| Multiple of 2 up until 1024
animation_earth_resolution = 64
show_animation = True
save_animation = True
plot_near_earth = True
# Particles ending up in 'region_of_interest' Earth-radia are interesting
region_of_interest = 3.0  # 640 km approximately

"""Particle settings"""
# Factors to initialize (relativistic) charged particle with
relativistic = False
simulate_anti_particles = False
mass_factor = 1.0
charge_factor = 1.0

"""Initial position settings"""
# Initial positions grid [m]
# Around 5E7 seems to give most interesting results ||| Custom grid spaces more points in outer grid coordinates
custom_grid = True
# Factor to adjust custom grid (lower is more towards edges, higher is more towards center)
scaling_factor = 0.069
# Approximately 0.0026 AU (=3.9E8 m) from Earth center
position_x = -3.9E8

particles_y = 180
minimum_y = -4.5E7
maximum_y = 4.5E7

particles_z = 180
minimum_z = -4.5E7
maximum_z = 4.5E7

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
            # Use custom spaced grid
            y_space = utils.custom_space(minimum_y, maximum_y, particles_y, scaling_factor)
            z_space = utils.custom_space(minimum_z, maximum_z, particles_z, scaling_factor)
        else:
            # Use linearly spaced grid
            y_space = np.linspace(minimum_y, maximum_y, particles_y)
            z_space = np.linspace(minimum_z, maximum_z, particles_z)

        # Define all initial velocities
        v_init_all = np.random.uniform(minimum_v, maximum_v, [particles_y, particles_z])

        for y in range(len(y_space)):
            # Initialize arrays for saving reduced simulation data
            particles_r = np.zeros((particles_z, save_data_points, 3))
            particles_v = np.zeros((particles_z, save_data_points, 3))

            # Whether the simulation over all Z-grid-coordinates for current Y-grid-coordinate was successful
            success = False

            # Keep retrying if simulation fails
            while not success:
                # Reference to multi-threading executor
                global executor
                # Storage for threads and their results
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
                    v_init = np.array([v_init_all[y, z], 0.0, 0.0])

                    # Change particles' charge for symmetry
                    charge_factor2 = 1
                    if simulate_anti_particles:
                        if position_y >= 0:
                            # Positrons instead of electrons
                            charge_factor2 = -1

                    # Simulate particle
                    if multi_threading:
                        # Multi-threaded simulation
                        future = executor.submit(sim.simulate, r_init, v_init, charge_factor*charge_factor2, mass_factor, dt, time_steps, region_of_interest, save_reduced, save_data_points, print_simulation_initialization, print_simulation_progress, z)
                        futures.append(future)
                    else:
                        # Single-threaded simulation
                        r_save_data, v_save_data, z2 = sim.simulate(r_init, v_init, charge_factor*charge_factor2, mass_factor, dt, time_steps, region_of_interest, save_reduced, save_data_points, print_simulation_initialization, print_simulation_progress, z)

                        particles_r[z, :, :] = r_save_data
                        particles_v[z, :, :] = v_save_data

                        success = True

                if multi_threading:
                    # Store multi-threaded simulation results
                    futures, _ = thread.wait(futures)
                    failed = False

                    for future in futures:
                        try:
                            r_save_data, v_save_data, z = future.result()
                        except Exception as exc:
                            # Rerun for this Y in case of failure... (no idea why this happens ||| probably Windows...)
                            print('Thread generated an exception: %s Retrying...' % exc)
                            # Create new multi-threading pool because old one is not usable any more
                            executor = thread.ProcessPoolExecutor(threads)
                            failed = True
                        else:
                            particles_r[z, :, :] = r_save_data
                            particles_v[z, :, :] = v_save_data
                    if not failed:
                        success = True

                # Save simulation data for every y (not all at once to avoid memory issues)
                if save_data and success:
                    # Save data for current y
                    file_str = 'Datasets/Data_t' + str(time) + 'dt' + str(dt) + \
                               'n' + str(particles_y * particles_z) + 'y' + str(y) + ".h5"
                    utils.create_datafile(file_str, particles_r, particles_v)

    # Particles absorption processing
    if do_post_processing:
        # Begin post-processing
        print("Post-processing...")
        savestring = 'relevant_data'

        
        cutoff_high = 1.1
        cutoff_low = 1.01
        
        if do_save_stripped_data:
            utils.save_relevant_data(savestring, cutoff_high, cutoff_low, particles_y)
        
        elif do_load_stripped_data:
            part_r, part_v, indices = utils.load_relevant_data(savestring, cutoff_high, cutoff_low, particles_y)

            
        # Load data
        # Filter useful trajectories (within 1.1-Earth Radius till 1.01-Earth-Radius)
        # Trace back trajectories to stop at absorption altitudes
        # Save differences in data (to get smaller file sizes...)
        else:
            pass
        pass

    # Data-processing for showing results
    if do_data_processing:
        # Begin data-processing
        print("Data-processing...")

        # Plot 3D Earth for static 3D plot
        fig, ax = utils.plot_earth(plot_simple, plot_earth_resolution)

        # Keep track of relevant particles amount
        relevant_count = 0
        # Keep track of relevant particles' grid coordinates
        relevant_xy = []

        # Iterate over Y-grid
        for y in range(particles_y):
            # Show progress
            if (y+1) % max(1, int(particles_y / 10)) == 0:
                print("Loading progress:", ("%.2f" % ((y+1) / particles_y * 100)), "%...")

            # Filename of dataset to load
            file_str = 'Datasets/Data_t' + str(time) + 'dt' + str(dt) + \
                       'n' + str(particles_y * particles_z) + 'y' + str(y) + ".h5"

            # Only load if dataset file exists
            if path.exists(file_str):
                # Load dataset
                r_dataset, v_dataset = utils.load_datafile(file_str)

                # Plot relevant trajectories
                for z in range(len(r_dataset)):
                    r_data = r_dataset[z]
                    v_data = v_dataset[z]
                    # Plot particle trajectory
                    if r_data[-1, 0] ** 2 + r_data[-1, 1] ** 2 + r_data[-1, 2] ** 2 < region_of_interest ** 2:
                        # Only plot when end-point is closer to than 3 Earth-radia (ignore deflected particles)
                        utils.plot_3d(ax, r_data, plot_near_earth)
                        # Add count of amount of relevant particles
                        relevant_count += 1
                        # Store Y- and Z-grid-coordinates
                        relevant_xy.append([y, z])
            else:
                print("File", file_str, "was not found...")

        print("Relevant particles: ", relevant_count, "out of", (particles_y*particles_z), "(",
              (relevant_count/particles_y/particles_z*100), "%)")

        # Initialize relevant particles array
        r_relevant = np.zeros([relevant_count, save_data_points, 3])

        # Retrieve relevant particles only from all particles
        for relevant_particle in range(relevant_count):
            # Retrieve Y- and Z-grid-coordinates
            y, z = relevant_xy[relevant_particle]
            # File-name of dataset to load
            file_str = 'Datasets/Data_t' + str(time) + 'dt' + str(dt) + \
                       'n' + str(particles_y * particles_z) + 'y' + str(y) + ".h5"
            # Load dataset using Y-grid-coordinate
            r_dataset, v_dataset = utils.load_datafile(file_str)
            # Load data using Z-grid-coordinate
            r_relevant[relevant_particle, :, :] = r_dataset[z]

        if show_animation:
            # Plot 3D Earth for animation
            fig, ax = utils.plot_earth(plot_simple, animation_earth_resolution)
            # Plot 3D particle trajectories with trail
            ani = utils.plot_3d_animation(fig, ax, r_relevant, plot_near_earth)
            if save_animation:
                # Define file name for animation
                file_str = 'Images/Animation_t' + str(time) + 'dt' + str(dt) + \
                       'n' + str(particles_y * particles_z) + ".gif"
                # Save animation
                utils.save_animation(file_str, ani)

    # End program timer
    time_elapsed = timeit.default_timer() - time_start
    print("Runtime:", time_elapsed, "seconds")
    print("Runtime:", time_elapsed / 60.0, "minutes")

    # Block program from terminating
    plt.show()

    return True

# Execute main program
if __name__ == '__main__':
    if main():
        exit()
