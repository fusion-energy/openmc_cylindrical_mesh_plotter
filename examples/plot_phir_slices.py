import numpy as np
import matplotlib.pyplot as plt
from math import pi

import openmc

mesh = openmc.CylindricalMesh()
mesh.phi_grid = np.linspace(0.0, 2 * pi, 10)
mesh.r_grid = np.linspace(0, 10, 4)
mesh.z_grid = np.linspace(0, 5, 5)

sp_filename= 'examples/statepoint.10.h5'
statepoint = openmc.StatePoint(sp_filename)

my_tally_result = statepoint.get_tally(name="my_tally")

actual = mesh.phi_grid

# Using linspace so that the endpoint of 360 is included
# actual = np.radians(np.linspace(0, 360, 20))
# expected = np.arange(0, 70, 10)
expected = mesh.r_grid
 
# r, theta = np.meshgrid(expected, actual)

# # values = np.random.random((actual.size, expected.size))
 
# fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
# ax.contourf(theta, r, values)

# plt.show()
