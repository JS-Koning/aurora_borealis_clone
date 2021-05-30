import numpy as np
import simulation as sim
import utilities as utils
import matplotlib.pyplot as plt

# Time between each iteration
dt = 1E-9
# Total time length [s]
time = 1E-3
# Time steps
time_steps = int(time/dt)

# Factors to initialize (relativistic) charged particle with
relativistic = True # TODO
mass_factor = 1.0
charge_factor = 2.0

# Particle grid
minimum_y = -3E8
maximum_y = 3E8
minimum_z = -3E8
maximum_z = 3E8

plot_simple = False
plot_near_earth = True


def main():
    r_init = np.array([-5E9, -5E7, 2E7])
    v_init = np.array([2000000.0, 0.0, 0.0])
    r_data, v_data = sim.simulate(r_init, v_init, charge_factor, mass_factor, dt, time_steps)

    return r_data, v_data


r, v = main()

#print(r)
#print(v)

r_i = r[0, :]
r_e = r[-1, :]
rr = r_e-r_i
distance = np.sqrt(rr[0]**2 + rr[1]**2 + rr[2]**2)

print("Particle moved a distance of: ", distance, "Earth-radius lengths.")
utils.plot_3d(r, plot_simple, plot_near_earth)

plt.show()

inx = utils.find_nearest(r, 1)
print(r[inx-1, :])
print(r[inx,:])
print(r[inx+1, :])



