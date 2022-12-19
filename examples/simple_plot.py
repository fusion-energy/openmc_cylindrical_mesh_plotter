# this example creates a simple CylindricalMesh tally and performs an openmc
# simulation to populate the tally. Slices of the resulting tally is then
# plotted using the openmc_cylindrical_plotter

import openmc
import numpy as np
from math import pi
from openmc_cylindrical_mesh_plotter import plot_rz_slice

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
my_source.space = openmc.stats.Point((0.,7., 1))
# my_source.angle = openmc.stats.Isotropic()
# my_source.energy = openmc.stats.Discrete([14e6], [1])

my_settings = openmc.Settings()
# my_settings.inactive = 0
my_settings.run_mode = "fixed source"
my_settings.batches = 10
my_settings.particles = 1000000
my_settings.source = my_source

model = openmc.model.Model(my_geometry, my_materials, my_settings, tallies)
sp_filename = model.run()

statepoint = openmc.StatePoint(sp_filename)

my_tally_result = statepoint.get_tally(name='my_tally')

mesh.write_data_to_vtk(
    filename="my_tally_result.vtk",
    datasets={"mean": my_tally_result.mean}
)

for slice_id in range(len(mesh.phi_grid)-1):
    plot  = plot_rz_slice(
        tally=my_tally_result,
        slice_id=slice_id
    )
    plot.show()
