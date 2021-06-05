import numpy as np
from numba import jit
import utilities as utils

"""Constants and parameters"""

# Radius of Earth [m]
# https://en.wikipedia.org/wiki/Earth_radius
# IUGG mean radius R_1
r_earth = 6371008.7714

# Mass of proton [kg]
# https://en.wikipedia.org/wiki/Proton
m_proton = 1.6726219236951E-27

# Mass of electron [kg]
# https://en.wikipedia.org/wiki/Electron
m_electron = 9.109383701528E-31

# Elementary (Coulomb) change [C]
# https://en.wikipedia.org/wiki/Elementary_charge
q_charge = 1.602176634E-19

# Earth's magnetic dipole tilt [rad]
# https://ase.tufts.edu/cosmos/view_picture.asp?id=326#:~:text=.&text=The%20magnetic%20axis%20is%20tilted,by%20the%20solar%20wind%20(Fig.
phi = -11.70 * np.pi / 180.0

# Earth's (mean) obliquity (axial tilt) [rad]
# https://en.wikipedia.org/wiki/Axial_tilt
theta = -23.4365472133 * np.pi / 180.0

# Earth's magnetic moment [A/m^2]
# https://www.tulane.edu/~sanelson/eens634/Hmwk6MagneticField.pdf
mu_earth = 7.94 * 10E22
mu = -mu_earth * np.array([np.sin(phi), 0.0, np.cos(phi)])

# Earth's dipole moment location (in 3D) [m, m, m]
# Set Earth's origin as center
r_dipole = np.array([0.0, 0.0, 0.0])

# Magnetic constant (vacuum permeability)
# https://en.wikipedia.org/wiki/Vacuum_permeability
mu_0 = np.pi * 4.0E-7


@jit(nopython=True)
def magnetic_field(r):
    # Get relative coordinate
    rr0 = np.array([r[0]-r_dipole[0], r[1]-r_dipole[1], r[2]-r_dipole[2]])*r_earth
    # Get relative distance
    r_magnetic = np.sqrt(rr0[0]**2 + rr0[1]**2 + rr0[2]**2)
    # Calculate magnetic field
    field = mu_0/(np.pi*4.0)*(3.0*rr0*np.dot(mu, rr0)/(r_magnetic**5)-mu/(r_magnetic**3))

    return field


@jit(nopython=True)
def runge_kutta_4(charge, mass, dt, r_particle_last_1, v_particle_last_1):
    # First order RK4
    a_particle_last_1 = charge/mass * np.cross(v_particle_last_1, magnetic_field(r_particle_last_1))

    # Second order RK4 (0.5)
    r_particle_last_2 = r_particle_last_1 + 0.5 * v_particle_last_1 * dt
    v_particle_last_2 = v_particle_last_1 + 0.5 * a_particle_last_1 * dt
    a_particle_last_2 = charge/mass * np.cross(v_particle_last_2, magnetic_field(r_particle_last_2))

    # Third order RK4 (0.5)
    r_particle_last_3 = r_particle_last_1 + 0.5 * v_particle_last_2 * dt
    v_particle_last_3 = v_particle_last_1 + 0.5 * a_particle_last_2 * dt
    a_particle_last_3 = charge / mass * np.cross(v_particle_last_3, magnetic_field(r_particle_last_3))

    # Fourth order RK4 (1.0)
    r_particle_last_4 = r_particle_last_1 + v_particle_last_3 * dt
    v_particle_last_4 = v_particle_last_1 + a_particle_last_3 * dt
    a_particle_last_4 = charge / mass * np.cross(v_particle_last_4, magnetic_field(r_particle_last_4))

    # New position and velocity
    r_particle = r_particle_last_1 + (dt / 6.0) *\
        (v_particle_last_1 + 2.0 * (v_particle_last_2 + v_particle_last_3) + v_particle_last_4)
    v_particle = v_particle_last_1 + (dt / 6.0) * \
        (a_particle_last_1 + 2.0 * (a_particle_last_2 + a_particle_last_3) + a_particle_last_4)

    return r_particle, v_particle


def simulate(r_particle_init, v_particle_init, charge, mass, dt, time_steps, region_of_interest, save_reduced, save_data_points, print_simulation_initialization, print_simulation_progress, current_id):
    # Initialize
    if print_simulation_initialization:
        print("Simulating single particle with [m =", mass, " q =", charge,
              " using Runge-Kutta-4 Algorithm...")

    r_particle = np.zeros((time_steps, 3))
    v_particle = np.zeros((time_steps, 3))

    # Initial condition
    r_particle[0, :] = r_particle_init / r_earth # now in m (only vary y,z => grid)
    v_particle[0, :] = v_particle_init # min 250, max 3000 km/s
    #v_particle[0, :], r_particle[0, :],  = utils.initialize_loc_vel(300000, 100, 1, 0)

    mass = m_electron * mass
    charge = -q_charge * charge
    
    # Simulate using algorithm
    for i in range(1, time_steps):
        if print_simulation_progress and (i+1) % int(time_steps / 10) == 0:
            print("Simulation progress:", ("%.2f" % ((i+1) / time_steps * 100)), "%...")

        # Runge-Kutta-4 algorithm
        r_particle[i, :], v_particle[i, :] = runge_kutta_4(charge, mass, dt, r_particle[i-1, :], v_particle[i-1, :])

        # Stop if at closest point and within RoI (Region of Interest)
        if r_particle[i, 0]**2 + r_particle[i, 1]**2 + r_particle[i, 2]**2 > r_particle[i-1, 0]**2 +\
                                                          r_particle[i-1, 1]**2 + r_particle[i-1, 2]**2 and r_particle[i, 0]**2 + r_particle[i, 1]**2 + r_particle[i, 2]**2 < region_of_interest**2:
        #    r_particle[i, :] = r_particle[i-1, :]
        #    v_particle[i, :] = v_particle[i-1, :]
            pass

    # Find index where simulation ends (choose index[3] to skip first 3 values as a failsafe)
    #index = np.where(r_particle[:-1, :] == r_particle[1:, :])[0]
    r_particle_radial = np.sqrt(r_particle[:,0]**2 + r_particle[:,1]**2 + r_particle[:,2]**2)
    index = [0, 0, 0, np.argmin(r_particle_radial)]

    if save_reduced:
        # Reduce simulation data
        if len(index) != 0:
            if index[3] >= save_data_points:
                # Save data until trajectory ends ||| Trajectory ended normally
                r_save_data = r_particle[max(0, index[3] - save_data_points):min(index[3], len(r_particle) - 1), :]
                v_save_data = v_particle[max(0, index[3] - save_data_points):min(index[3], len(v_particle) - 1), :]
            else:
                # Save data from beginning of simulation ||| Trajectory ended too soon
                r_save_data = r_particle[0: save_data_points, :]
                v_save_data = v_particle[0: save_data_points, :]
        else:
            # Save data from last part of simulation ||| Trajectory did not end (yet)
            r_save_data = r_particle[len(r_particle) - 1 - save_data_points:len(r_particle) - 1, :]
            v_save_data = v_particle[len(v_particle) - 1 - save_data_points:len(v_particle) - 1, :]

        return r_save_data, v_save_data, current_id
    else:
        return  r_particle, v_particle, current_id


def incoming_probabilities(p_electron, p_proton, p_alpha, partnums):

    masses = np.array([m_electron, m_proton, 4*m_proton])
    charges = np.array([-1, 1, 2]) *q_charge
    
    rands = np.random.rand(partnums)
    masses_arr = np.zeros(partnums)
    charges_arr = np.zeros(partnums)
    for i in range(len(masses_arr)):
        if rands[i] < p_proton:
            masses_arr[i] = masses[1]
            charges_arr[i] = charges[1]
        elif rands[i] > 1-p_alpha:
            masses_arr[i] = masses[2]
            charges_arr[i] = charges[2]
        else:
            masses_arr[i] = masses[0]
            charges_arr[i] = charges[0]
    return masses_arr, charges_arr
