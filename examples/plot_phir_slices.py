# this example creates a simple CylindricalMesh tally and performs an openmc
# simulation to populate the tally. Slices of the resulting tally is then
# plotted using the openmc_cylindrical_plotter

import openmc
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import openmc_cylindrical_mesh_plotter  # adds slice_of_data method to CylindricalMesh

mesh = openmc.CylindricalMesh()
mesh.phi_grid = np.linspace(0.0, 2 * pi, 10)
mesh.r_grid = np.linspace(0, 10, 20)
mesh.z_grid = np.linspace(0, 5, 4)

tally = openmc.Tally(name="my_tally")
mesh_filter = openmc.MeshFilter(mesh)
tally.filters.append(mesh_filter)
tally.scores.append("flux")
tallies = openmc.Tallies([tally])

outer_surface = openmc.Sphere(r=100, boundary_type="vacuum")
cell = openmc.Cell(region=-outer_surface)

material = openmc.Material()
material.add_nuclide("Fe56", 1)
material.set_density("g/cm3", 0.1)
my_materials = openmc.Materials([material])

universe = openmc.Universe(cells=[cell])
my_geometry = openmc.Geometry(universe)

my_source = openmc.Source()

# the distribution of radius is just a single value
radius = openmc.stats.Discrete([5], [1])
# the distribution of source z values is just a single value
z_values = openmc.stats.Discrete([2.5], [1])
# the distribution of source azimuthal angles values is a uniform distribution between 0 and 2 Pi
angle = openmc.stats.Uniform(a=0.0, b=1 * 3.14159265359)
# this makes the ring source using the three distributions and a radius
# could do a point source instead with
# my_source.space = openmc.stats.Point((0,0,0))
my_source.space = openmc.stats.CylindricalIndependent(
    r=radius, phi=angle, z=z_values, origin=(0.0, 0.0, 0.0)
)
# sets the direction to isotropic
my_source.angle = openmc.stats.Isotropic()


my_settings = openmc.Settings()
# my_settings.inactive = 0
my_settings.run_mode = "fixed source"
my_settings.batches = 10
my_settings.particles = 100000
my_settings.source = my_source

model = openmc.model.Model(my_geometry, my_materials, my_settings, tallies)
sp_filename = model.run()

statepoint = openmc.StatePoint(sp_filename)

my_tally_result = statepoint.get_tally(name="my_tally")

actual = np.linspace(mesh.phi_grid[0],mesh.phi_grid[-1], mesh.dimension[1])
expected = np.linspace(mesh.r_grid[0],mesh.r_grid[-1], mesh.dimension[0])

# Using linspace so that the endpoint of 360 is included
# actual = np.radians(np.linspace(0, 360, 20))
# expected = np.arange(0, 70, 10)

r, theta = np.meshgrid(expected, actual)
values=np.array([(len(mesh.r_grid)-1)*[12]]*(len(mesh.phi_grid)-1))
# # values = np.random.random((actual.size, expected.size))

for slice_index in range(len(mesh.z_grid)-1):
    lower_index = int(slice_index*(len(mesh.phi_grid)-1))
    upper_index = int((slice_index+1)*(len(mesh.phi_grid)-1))

    dataset=my_tally_result.mean


    # values=dataset.flatten().reshape(-1,len(mesh.r_grid)-1,order='C')[:len(mesh.phi_grid)-1]
    values=dataset.flatten().reshape(-1,len(mesh.r_grid)-1,order='A')[lower_index:upper_index]
    # rot_val = np.rot90(values)

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    from matplotlib import ticker
    im = ax.contourf(theta, r, values)#, locator=ticker.LogLocator())

    # ax.contour(theta, r, theta)
    # ax.contourf(theta, r, data_slice)

    plt.colorbar(im)
    plt.show()
