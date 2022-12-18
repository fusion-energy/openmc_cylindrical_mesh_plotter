import matplotlib.pyplot as plt
import numpy as np

import openmc
from math import pi

mesh = openmc.CylindricalMesh()
mesh.r_grid = np.linspace(2,10,50)
mesh.phi_grid = np.linspace(0.,2*pi,10)
mesh.z_grid = np.linspace(0,5,40)


tally = openmc.Tally(name='my_tally')
mesh_filter = openmc.MeshFilter(mesh)
tally.filters.append(mesh_filter)
tally.scores.append("flux")
tallies = openmc.Tallies([tally])

outer_surface = openmc.Sphere(r=100, boundary_type='vacuum')
cell = openmc.Cell(region = -outer_surface)

material = openmc.Material()
material.add_element('Fe', 1)
material.set_density('g/cm3', 1)
my_materials = openmc.Materials([material])

universe = openmc.Universe(cells=[cell])
my_geometry = openmc.Geometry(universe)

my_source = openmc.Source()
my_source.space = openmc.stats.Point((0., 0., 1))
my_source.angle = openmc.stats.Isotropic()
my_source.energy = openmc.stats.Discrete([14e6], [1])

my_settings = openmc.Settings()
my_settings.inactive = 0
my_settings.run_mode = "fixed source"
my_settings.batches = 10
my_settings.particles = 100000
my_settings.source = my_source

model = openmc.model.Model(my_geometry, my_materials, my_settings, tallies)
sp_filename = model.run()

statepoint = openmc.StatePoint(sp_filename)

my_tally_result = statepoint.get_tally(name='my_tally')

mesh.write_data_to_vtk(
    filename="my_tally_result.vtk",
    datasets={"mean": my_tally_result.mean}
)

reshaped_using_dim=my_tally_result.mean.reshape(mesh.dimension)

xmin = mesh.r_grid[0]
xmax = mesh.r_grid[-1]
ymin = mesh.z_grid[0]
ymax = mesh.z_grid[-1]
plt.imshow(reshaped_using_dim[1], extent=[xmin,xmax,ymin,ymax])
plt.xlabel('X [cm]')
plt.ylabel('Y [cm]')
plt.show()

reshaped=my_tally_result.mean.reshape((mesh.dimension[2], mesh.dimension[1], mesh.dimension[0]))

xmin = mesh.r_grid[0]
xmax = mesh.r_grid[-1]
ymin = mesh.z_grid[0]
ymax = mesh.z_grid[-1]
plt.imshow(reshaped[1], extent=[xmin,xmax,ymin,ymax])
plt.xlabel('X [cm]')
plt.ylabel('Y [cm]')
plt.show()
