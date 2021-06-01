import numpy as np
import simulation as sim
import utilities as utils
import matplotlib.pyplot as plt
import timeit

time_start = timeit.default_timer()

# Seed for reproducibility
seed = 420
np.random.seed(seed)

# Time between each iteration # minimum: 1E-6 | okay: 1E-7 | good: 1E-8 | best: 1E-9
dt = 1E-8
# Total time length [s]
time = 1E-2
# Time steps
time_steps = int(time/dt)

# Factors to initialize (relativistic) charged particle with
relativistic = False #  TODO
mass_factor = 1.0
charge_factor = 1.0

# Initial positions grid [m]
position_x = -1E10

particles_y = 50
minimum_y = -1E8
maximum_y = 1E8

particles_z = 50
minimum_z = -1E8
maximum_z = 1E8

# Initial velocities [m/s]
minimum_v = 2.5e5
maximum_v = 3.0e6
maximum_v = 7.5e5

# Plot settings
plot_simple = False
plot_near_earth = True
plot_points = 2000


def main():
    # Plot Earth
    fig, ax = utils.plot_earth(plot_simple)

    y_space = np.linspace(minimum_y,maximum_y,particles_y)
    z_space = np.linspace(minimum_z,maximum_z,particles_z)

    mass, charge = sim.incoming_probabilities(0.95, 0.05*0.95, 0.05**2, particles_z*particles_y)

    for y in range(len(y_space)):
        for z in range(len(z_space)):
            print("Simulating particle:", y*particles_y + z + 1, "out of", particles_y*particles_z)

            # Define grid positions
            position_y = y_space[y]
            position_z = z_space[z]

            # Initialize particle position
            r_init = np.array([position_x, position_y, position_z])
            # Initialize particle velocity
            v_init = np.array([np.random.normal((maximum_v+minimum_v)/2,(maximum_v+minimum_v)/10), 0.0, 0.0])

            # Change particles's charge for symmetry
            charge_factor2 = 1
            if position_y >= 0:
                charge_factor2 = -1

            # Simulate particle
            r_data, v_data = sim.simulate(r_init, v_init, charge_factor*charge_factor2, mass_factor, dt, time_steps)

            # Plot particle trajectory
            if r_data[-1, 0]**2 + r_data[-1, 1]**2 + r_data[-1, 2]**2 < 3**2:
                # Only plot when end-point is closer to than 3 Earth-radia (ignore deflected particles)
                utils.plot_3d(ax, r_data, plot_near_earth, plot_points)

    return r_data, v_data


r, v = main()

time_elapsed = (timeit.default_timer() - time_start)
print(time_elapsed)

r_i = r[0, :]
r_e = r[-1, :]
rr = r_e-r_i
distance = np.sqrt(rr[0]**2 + rr[1]**2 + rr[2]**2)

print("Particle moved a distance of: ", distance, "Earth-radius lengths.")

plt.show()

inx = utils.find_nearest(r, 1)
print(r[inx-1, :])
print(r[inx,:])
print(r[inx+1, :])


