import numpy as np
import simulation as sim

# Time between each iteration
dt = 0.001
# Total time length [s]
time = 5000
# Time steps
time_steps = int(time/dt)

# Factor to initialize charged particle with
mass_factor = 4.0
charge_factor = 2.0


def main():
    sim.simulate(charge_factor, mass_factor, dt, time_steps)


main()