import numpy as np

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
phi = 11.70 * np.pi / 180.0

# Earth's (mean) obliquity (axial tilt) [rad]
# https://en.wikipedia.org/wiki/Axial_tilt
theta = 23.4365472133 * np.pi / 180.0

# Earth's magnetic moment [A/m^2]
# https://www.tulane.edu/~sanelson/eens634/Hmwk6MagneticField.pdf
mu_earth = 7.94 * 10E22
mu = -mu_earth * np.array([0.0, np.sin(phi), np.cos(phi)])

# Earth's dipole moment location (in 3D) [m, m, m]
# Set Earth's origin as center
r_dipole = np.array([0.0, 0.0, 0.0])

# Magnetic constant (vacuum permeability)
# https://en.wikipedia.org/wiki/Vacuum_permeability
mu_0 = np.pi * 4.0E-7


def magnetic_field(r, r0):
    # Get relative coordinate
    rr0 = np.array([r[0]-r0[0], r[1]-r0[1], r[2]-r0[2]])
    # Get relative distance
    r_magnetic = np.sqrt(rr0[0]**2 + rr0[1]**2 + rr0[2]**2)
    # Calculate magnetic field
    field = mu_0*(3.0*rr0*np.dot(mu, rr0)/(r_magnetic**5)-mu/(r_magnetic**3))
    return field


def runge_kutta_4(charge, mass, dt, r_particle_last_1, v_particle_last_1):
    # First order RK4
    a_particle_last_1 = charge/mass * np.cross(v_particle_last_1, magnetic_field(r_particle_last_1, r_dipole))

    # Second order RK4 (0.5)
    r_particle_last_2 = r_particle_last_1 + 0.5 * v_particle_last_1 * dt
    v_particle_last_2 = v_particle_last_1 + 0.5 * a_particle_last_1 * dt
    a_particle_last_2 = charge/mass * np.cross(v_particle_last_2, magnetic_field(r_particle_last_2, r_dipole))

    # Third order RK4 (0.5)
    r_particle_last_3 = r_particle_last_2 + 0.5 * v_particle_last_2 * dt
    v_particle_last_3 = v_particle_last_2 + 0.5 * a_particle_last_2 * dt
    a_particle_last_3 = charge / mass * np.cross(v_particle_last_3, magnetic_field(r_particle_last_3, r_dipole))

    # Fourth order RK4 (1.0)
    r_particle_last_4 = r_particle_last_3 + v_particle_last_3 * dt
    v_particle_last_4 = v_particle_last_3 + a_particle_last_3 * dt
    a_particle_last_4 = charge / mass * np.cross(v_particle_last_4, magnetic_field(r_particle_last_4, r_dipole))

    # New position and velocity
    r_particle = r_particle_last_1 + (dt / 6.0) *\
        (v_particle_last_1 + 2.0 * (v_particle_last_2 + v_particle_last_3) + v_particle_last_4)
    v_particle = v_particle_last_1 + (dt / 6.0) * \
        (a_particle_last_1 + 2.0 * (a_particle_last_2 + a_particle_last_3) + a_particle_last_4)

    return r_particle, v_particle


def simulate(charge_factor, mass_factor, dt, time_steps):
    # Initialize
    print("Simulating single particle with [m =", mass_factor, "* proton mass, q =", charge_factor,
          "* elementary charge] using Runge-Kutta-4 Algorithm...")
    r_particle = np.zeros((time_steps, 3))
    v_particle = np.zeros((time_steps, 3))

    # Charged particle
    m = mass_factor * m_proton
    q = charge_factor * q_charge

    # Initial condition
    r_particle[0, :] = np.array([-30.0, -30.0, -30.0])
    v_particle[0, :] = np.array([-2.0, -1.0, 10.0])

    # Simulate using algorithm
    for i in range(1, time_steps):
        if i % int(time_steps / 100) == 0:
            print("Progress:", ("%.2f" % (i / time_steps * 100)), "%...")

        # Runge-Kutta-4 algorithm
        r_particle[i, :], v_particle[i, :] = runge_kutta_4(q, m, dt, r_particle[i-1, :], v_particle[i-1, :])

    return r_particle, v_particle
