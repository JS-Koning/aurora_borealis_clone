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
r_dipole = np.zeros([0.0, 0.0, 0.0])

# Magnetic constant (vacuum permeability)
# https://en.wikipedia.org/wiki/Vacuum_permeability
mu_0 = np.pi * 4.0E-7


def magnetic_field(r, r0, mu):
    return 0