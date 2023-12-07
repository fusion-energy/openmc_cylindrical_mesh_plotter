# this example creates a simple CylindricalMesh tally and performs an openmc
# simulation to populate the tally. Slices of the resulting tally is then
# plotted using the openmc_cylindrical_plotter

import openmc
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from openmc_cylindrical_mesh_plotter import plot_mesh_tally_rz_slice
from matplotlib.colors import LogNorm

mesh = openmc.CylindricalMesh(
    phi_grid=np.linspace(0.0, 2 * pi, 3),
    r_grid=np.linspace(0, 10, 20),
    z_grid=np.linspace(0, 5, 10),
)

tally = openmc.Tally(name="my_tally")
mesh_filter = openmc.MeshFilter(mesh)
tally.filters.append(mesh_filter)
tally.scores.append("flux")
tallies = openmc.Tallies([tally])

material = openmc.Material()
material.add_nuclide("Fe56", 1)
material.set_density("g/cm3", 0.1)

my_materials = openmc.Materials([material])
outer_surface = openmc.Sphere(r=100, boundary_type="vacuum")
cell = openmc.Cell(region=-outer_surface, fill=material)


my_geometry = openmc.Geometry([cell])


my_source = openmc.IndependentSource()
my_source.space = openmc.stats.Point((0, 0, 0))
my_source.angle = openmc.stats.Isotropic()
my_source.energy = openmc.stats.Discrete([14e6], [1])


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

for slice_index in range(0, len(mesh.phi_grid) - 1):
    plot = plot_mesh_tally_rz_slice(
        tally=my_tally_result,
        outline=True,
        geometry=my_geometry,
        norm=LogNorm(),
        slice_index=slice_index,
    )
    plot.figure.savefig(f"rz_point_source_{slice_index}.png")
    print(f"written rz_point_source_{slice_index}.png")
