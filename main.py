import numpy as np
import simulation as sim
import utilities as utils
import matplotlib.pyplot as plt

# Time between each iteration
dt = 0.0001
# Total time length [s]
time = 5000
# Time steps
time_steps = int(time/dt)

# Factor to initialize charged particle with
mass_factor = 4.0
charge_factor = 2.0


def main():
    r_data, v_data = sim.simulate(charge_factor, mass_factor, dt, time_steps)

    return r_data, v_data


r, v = main()

r_i = r[0, :]
r_e = r[-1, :]
rr = r_e-r_i
distance = np.sqrt(rr[0]**2 + rr[1]**2 + rr[2]**2)

print("Particle moved a distance of: ", distance, "Earth-radius lengths.")
utils.plot_3d(r)

plt.show()