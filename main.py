"""Import settings"""
import numpy as np
import simulation as sim
import utilities as utils
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
import timeit
import concurrent.futures as thread
import psutil
from os import path
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

"""Program settings"""
# Initial trajectory simulation
do_simulation = True
# Particle absorption simulation
do_post_processing = True
# Data processing
do_data_processing = True

"""Logging settings"""
print_simulation_progress = False
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
# Apply seed to random functions within NumPy
np.random.seed(seed)

"""Time settings"""
# Time between each iteration ||| minimum: 1E-6 ||| okay: 1E-7 ||| good: 1E-8 ||| best: 1E-9
dt = 1E-9
# Total time length [s] ||| direct only: at least 1E-3 ||| indirect and direct: at least 5E-1
time = 1E-3
# Time steps
time_steps = int(time/dt)

"""Dataset settings"""
# Should simulation datasets be saved
save_data = True
# Should simulation datasets be reduced before saving
save_reduced = True
# Reduced dataset size (trajectory points)
save_data_points = round(3 * 1E-5/dt)

"""Post-processing settings"""
# Should the relevant particles be saved
do_save_stripped_data = False
# Should the relevant particles be loaded
do_load_stripped_data = True
# Should aurora data be generated
do_create_aurora = False
# Should errorbars be created
do_create_errorbars = True
# What is the upper bound for auroras (in R_Earth) ||| Approximately 640 km
relevant_upper_bound_altitude = 1.1
# What is the lower bound for auroras (in R_Earth) ||| Approximately 64 km
relevant_lower_bound_altitude = 1.01

"""Data-processing settings"""
# Use stripped data (speeds up)
use_stripped_data = True

"""Plot settings"""
# Should a simple Earth model be drawn instead of a (heavier) realistic one
plot_simple = False
# Should plots and animations be rendered near Earth
plot_near_earth = True
# Which trajectories should be plotted ||| Particles ending up in approximately 640km (1.1 Earth-radia) are interesting
plot_region_of_interest = 1.1
# Resolution of Earth texture ||| Multiple of 2 up until 1024
plot_earth_resolution = 64
# Should animations be shown
show_animation = False
# Resolution of Earth texture ||| Multiple of 2 up until 1024
animation_earth_resolution = 64
# Should animations be saved after being shown
save_animation = True
# Should aurora be shown
show_aurora = False

"""Particle settings"""
# Factors to initialize (relativistic) charged particle with || UNUSED
relativistic = False
# Should anti-particles be simulated (not realistic) || USED
simulate_anti_particles = False
# Relativistic mass factor || SEMI-USED
mass_factor = 1.0
# Relativistic charge factor || SEMI-USED
charge_factor = 1.0

"""Initial position settings"""
# Around 5E7 seems to give most interesting results ||| Custom grid spaces more points in outer grid coordinates
custom_grid = True
# Factor to adjust custom grid (lower is more towards edges, higher is more towards center)
scaling_factor = 0.069

# Initial positions grid in y,z plane at x distance [m]
# X-grid settings
# Approximately 0.0026 AU (=3.9E8 m) from Earth center ||| More than 60 Earth-radia away
position_x = -3.9E8

# Y-grid settings
# Amount of particles in Y-direction
particles_y = 180
# Minimal Y position
minimum_y = -4.5E7
# Maximal Y position
maximum_y = 4.5E7

# Z-grid settings
# Amount of particles in Z-direction
particles_z = 180
# Minimal Z position
minimum_z = -4.5E7
# Maximal Z position
maximum_z = 4.5E7

"""Initial velocity settings"""
# Initial velocities (uniform distribution) [m/s] ||| https://link.springer.com/article/10.1007/s11207-020-01730-z
# minimum velocity: 300 km/s ||| Slow solar wind @ 0.9 ~ 1.1 AU
minimum_v = 3.0e5
# maximum velocity: 750 km/s ||| Fast solar wind @ 0.9 ~ 1.1 AU
maximum_v = 7.5e5
# Plasma wave interaction contribution
simulate_plasma_wave_interaction = True
# Auroral acceleration region (https://www.nature.com/articles/s41467-021-23377-5)
acceleration_region = 3.0  # 3 Earth-radia
# How fast should the particles be sped up towards the target energy ||| Lower = faster; higher = slower
interpolation_strength = 4.0
# Acceleration target energy (parameters for Gamma/Maxwell distribution)
# https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JA073i007p02325
# http://adsabs.harvard.edu/full/1969SSRv...10..413R
# https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JA076i001p00063
# https://link.springer.com/content/pdf/10.1186/BF03352005.pdf
peak_target_energy = 1400 * 3.0  # keV
gamma_target_energy = 4.3 * 1000.0


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

        # Define all velocity factors using Gamma/Maxwell distribution
        # https://link.springer.com/content/pdf/10.1186/BF03352005.pdf
        target_energy_all = np.random.gamma(peak_target_energy / gamma_target_energy,
                                            gamma_target_energy, [particles_y, particles_z])

        velocity_factor_all = np.sqrt(target_energy_all / (0.5 * sim.m_electron / sim.q_charge))
        velocity_factor_all /= (minimum_v + maximum_v) / 2.0

        for y in range(len(y_space)):
            # Initialize arrays for saving reduced simulation data
            if save_reduced:
                particles_r = np.zeros((particles_z, save_data_points, 3))
                particles_v = np.zeros((particles_z, save_data_points, 3))
            else:
                particles_r = np.zeros((particles_z, time_steps, 3))
                particles_v = np.zeros((particles_z, time_steps, 3))

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
                    # Initialize velocity factor
                    if simulate_plasma_wave_interaction:
                        velocity_factor = velocity_factor_all[y, z]
                    else:
                        velocity_factor = 1.0

                    # Change particles' charge for symmetry
                    charge_factor2 = 1
                    if simulate_anti_particles:
                        if position_y >= 0:
                            # Positrons instead of electrons
                            charge_factor2 = -1

                    # Simulate particle
                    if multi_threading:
                        # Multi-threaded simulation
                        future = executor.submit(sim.simulate, r_init, v_init, charge_factor*charge_factor2,
                                                 mass_factor, dt, time_steps, acceleration_region,
                                                 velocity_factor, interpolation_strength, save_reduced,
                                                 save_data_points, print_simulation_initialization,
                                                 print_simulation_progress, z)
                        futures.append(future)
                    else:
                        # Single-threaded simulation
                        r_save_data, v_save_data, z2 = sim.simulate(r_init, v_init, charge_factor*charge_factor2,
                                                                    mass_factor, dt, time_steps, acceleration_region,
                                                                    velocity_factor, interpolation_strength,
                                                                    save_reduced, save_data_points,
                                                                    print_simulation_initialization,
                                                                    print_simulation_progress, z)

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

        if do_save_stripped_data:
            print("Saving stripped data...")

            if save_reduced:
                utils.save_relevant_data(relevant_upper_bound_altitude, relevant_lower_bound_altitude, particles_y,
                                         time, dt, particles_y*particles_z, save_data_points)
            else:
                utils.save_relevant_data(relevant_upper_bound_altitude, relevant_lower_bound_altitude, particles_y,
                                         time, dt, particles_y * particles_z, time_steps)

        if do_load_stripped_data:
            print("Loading stripped data...")

            file_str = 'Datasets/DataStripped_t' + str(time) + 'dt' + str(dt) + 'n' \
                       + str(particles_y*particles_z) + ".h5"
            if path.exists(file_str):
                part_r, part_v, indices = utils.load_relevant_data(file_str)
                indices = indices.astype(int)
                # distances = np.linalg.norm(part_r, axis=2)
                velocities = np.linalg.norm(part_v, axis=2)
                energies = 0.5 * sim.m_electron * velocities**2 / (sim.q_charge*1000)  # in keV

                if do_create_aurora:
                    file_str_new = 'Datasets/DataProcessed_t' + str(time) + 'dt' + str(dt) + 'n' \
                               + str(particles_y * particles_z) + ".h5"

                    height_locs = utils.gasses_absorption(energies)
                    print("Lowest height:", np.min(height_locs), "; Heighest height:", np.max(height_locs))
                    xyz = utils.location_absorption(part_r, height_locs, indices)
                    print("Lowest height:", np.min(np.linalg.norm(xyz, axis=1)), "; Heighest height:",
                          np.max(np.linalg.norm(xyz, axis=1)))

                    part_r_new, part_v_new = utils.post_process(part_r, part_v, xyz)
                    utils.create_datafile(file_str_new, part_r_new, part_v_new)

                    print("Created post-processed datasets")
                if do_create_errorbars:
                    iterations = 100
                    xyz_iterated = np.zeros((iterations, len(energies), 3))
                    for k in range(iterations):
                        xyz_iterated[k, :, :] = utils.location_absorption(part_r, utils.gasses_absorption(energies), indices)
                    distances_errorbar = np.linalg.norm(xyz_iterated, axis=2)
                    errorbar = np.std(distances_errorbar, axis=0)
                    print(errorbar.shape)
                    print(distances_errorbar.shape)
                    print(energies.shape)
                    utils.create_plot_errorbar(energies[:, -1], 'errorbar of absorption energy', 'Energy', 'Height', y_data=np.average(distances_errorbar, axis=0), error_bar=errorbar)

            else:
                print("Could not find stripped dataset...")

    # Data-processing for showing results
    if do_data_processing:
        # Begin data-processing
        print("Data-processing...")

        # Plot 3D Earth for static 3D plot
        fig, ax = utils.plot_earth(plot_simple, plot_earth_resolution)

        if use_stripped_data:
            file_str = 'Datasets/DataStripped_t' + str(time) + 'dt' + str(dt) + 'n' \
                       + str(particles_y * particles_z) + ".h5"
            if path.exists(file_str):
                r_relevant, v_relevant, indices = utils.load_relevant_data(file_str)
                relevant_count = len(r_relevant)
                indices = indices.astype(int)
                for z in range(len(r_relevant)):
                    r_data = r_relevant[z]
                    # Plot particle trajectory
                    if r_data[-1, 0] ** 2 + r_data[-1, 1] ** 2 + r_data[-1, 2] ** 2 < plot_region_of_interest ** 2:
                        # Only plot when end-point is closer to than 3 Earth-radia (ignore deflected particles)
                        utils.plot_3d(ax, r_data, plot_near_earth)
            else:
                print("Stripped dataset not found")
        else:
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
                        # v_data = v_dataset[z]

                        # Plot particle trajectory
                        if r_data[-1, 0] ** 2 + r_data[-1, 1] ** 2 + r_data[-1, 2] ** 2 < plot_region_of_interest ** 2:
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
            if save_reduced:
                r_relevant = np.zeros([relevant_count, save_data_points, 3])
                v_relevant = np.zeros([relevant_count, save_data_points, 3])
            else:
                r_relevant = np.zeros([relevant_count, time_steps, 3])
                v_relevant = np.zeros([relevant_count, time_steps, 3])

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
                v_relevant[relevant_particle, :, :] = v_dataset[z]

        if relevant_count != 0:
            # distances = np.linalg.norm(r_relevant, axis=2)
            velocities = np.linalg.norm(v_relevant, axis=2)
            plt.figure()
            for i in range(len(velocities[:, 0])):
                plt.plot(velocities[i, :])
            plt.xlabel("Time [$ns$]")
            plt.ylabel("Velocity [$\\frac{m}{s}$]")
            plt.show(block=False)
            print("Mean velocity", np.mean(velocities[:, -1]))
            print("Max velocity", np.max(velocities[:, -1]))
            print("Min velocity", np.min(velocities[:, -1]))

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

            file_str_new = 'Datasets/DataProcessed_t' + str(time) + 'dt' + str(dt) + 'n' \
                           + str(particles_y * particles_z) + ".h5"
            r_dataset_new, v_dataset_new = utils.load_datafile(file_str_new)
            # Plot 3D Earth for animation
            fig, ax = utils.plot_earth(plot_simple, animation_earth_resolution)
            # Plot 3D particle trajectories with trail
            ani = utils.plot_3d_animation(fig, ax, r_dataset_new, plot_near_earth)
            if save_animation:
                # Define file name for animation
                file_str = 'Images/AnimationProcessed_t' + str(time) + 'dt' + str(dt) + \
                           'n' + str(particles_y * particles_z) + ".gif"
                # Save animation
                utils.save_animation(file_str, ani)

        if show_aurora:
            file_str_new = 'Datasets/DataProcessed_t' + str(time) + 'dt' + str(dt) + 'n' \
                           + str(particles_y * particles_z) + ".h5"
            r_dataset_new, v_dataset_new = utils.load_datafile(file_str_new)

            velocities_full = np.linalg.norm(v_dataset_new, axis=2)
            velocities = np.max(np.linalg.norm(v_dataset_new, axis=2), axis=1)
            heights = np.linalg.norm(r_dataset_new[:, -1, :], axis=1)
            altitudes = (heights - 1.0) * sim.r_earth / 1000  # in km
            energies = 0.5 * sim.m_electron * velocities ** 2 / (sim.q_charge * 1000)  # in keV

            # altitude_up = (relevant_upper_bound_altitude-1.0) * sim.r_earth / 1000
            # altitude_down = (relevant_lower_bound_altitude - 1.0) * sim.r_earth / 1000
            altitude_up = np.max(altitudes)
            altitude_down = np.min(altitudes)

            # plotting the particle interactions
            fig, ax = utils.plot_earth(plot_simple, plot_earth_resolution)
            ax.scatter(r_dataset_new[:, -1, 0], r_dataset_new[:, -1, 1], r_dataset_new[:, -1, 2], c="green")
            plt.show(block=False)

            # plotting the particle interactions with trajectories
            fig, ax = utils.plot_earth(plot_simple, plot_earth_resolution)
            for z in range(len(r_dataset_new)):
                r_data = r_dataset_new[z]
                # Plot particle trajectory
                utils.plot_3d(ax, r_data, plot_near_earth)
            ax.scatter(r_dataset_new[:, -1, 0], r_dataset_new[:, -1, 1], r_dataset_new[:, -1, 2], c="green")

            plt.show(block=False)

            print("Average altitude:", np.mean(altitudes))

            # velocity vs. aurora height
            plt.figure()
            plt.scatter(velocities, heights)
            plt.ylabel("Distance from Earth's center [$R_{Earth}$]")
            plt.xlabel("Velocity [$\\frac{m}{s}$]")
            plt.show(block=False)

            # velocity vs. time
            plt.figure()
            for i in range(len(velocities_full[:, 0])):
                plt.plot(velocities_full[i, :])
            plt.xlabel("Time [$ns$]")
            plt.ylabel("Velocity [$\\frac{m}{s}$]")
            plt.show(block=False)

            # energies vs. altitude
            plt.figure()
            for i in range(len(altitudes)):
                # uniform distribution
                color_factor = (altitudes[i]-altitude_down)/(altitude_up-altitude_down)
                # push green to lower altitude
                color_factor = np.power(color_factor, 0.45)

                # color = plt.cm.jet(color_factor)
                color = plt.cm.rainbow(color_factor)
                # color = plt.cm.turbo(color_factor)
                # color = plt.cm.nipy_spectral(color_factor)

                # Darken colors
                color = colorsys.rgb_to_hls(*mc.to_rgb(color))
                # 1.5 = darker, 0.5 = lighter
                color = colorsys.hls_to_rgb(color[0], 1 - 1.05 * (1 - color[1]), color[2])

                plt.scatter(energies[i], altitudes[i], c=color)

            plt.ylabel("Altitude [$km$]")
            plt.xlabel("Energy [$keV$]")
            plt.show(block=False)

            plt.figure()
            # per 20 km-ish
            plt.hist(altitudes, orientation='horizontal', bins=29)
            plt.ylabel("Altitude [$km$]")
            plt.xlabel("Amount of particles [#]")
            plt.show(block=False)

            # plotting the particle interactions (fancy)
            fig, ax = utils.plot_earth(plot_simple, plot_earth_resolution)
            for z in range(len(r_dataset_new)):
                r_data = r_dataset_new[z]
                # uniform distribution
                color_factor = (altitudes[z] - altitude_down) / (altitude_up - altitude_down)
                # push green to lower altitude
                color_factor = np.power(color_factor, 0.45)

                # color = plt.cm.jet(color_factor)
                color = plt.cm.rainbow(color_factor)
                # color = plt.cm.turbo(color_factor)
                # color = plt.cm.nipy_spectral(color_factor)

                # Darken colors
                color = colorsys.rgb_to_hls(*mc.to_rgb(color))
                # 1.5 = darker, 0.5 = lighter
                color = colorsys.hls_to_rgb(color[0], 1 - 1.05 * (1 - color[1]), color[2])
                ax.scatter(r_data[-1, 0], r_data[-1, 1], r_data[-1, 2], c=color)

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
