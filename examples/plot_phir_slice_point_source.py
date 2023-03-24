# this example creates a simple CylindricalMesh tally and performs an openmc
# simulation to populate the tally. Slices of the resulting tally is then
# plotted using the openmc_cylindrical_plotter in the Phi R axis.

import openmc
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import openmc_cylindrical_mesh_plotter  # adds slice_of_data method to CylindricalMesh
from matplotlib import ticker

mesh = openmc.CylindricalMesh()
mesh.phi_grid = np.linspace(0, 2 * pi, 50)  # note the mesh is full 2pi circle
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

# this makes a point source instead with
my_source.space = openmc.stats.Point((0, -5, 0))
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

for slice_index in range(1, len(mesh.z_grid)):
    theta, r, values = mesh.slice_of_data(
        dataset=my_tally_result.mean.flatten(),
        slice_index=slice_index,
        axis="PhiR",
        volume_normalization=False,
    )

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    im = ax.contourf(
        theta, r, values, extent=(0, 100, 0, 50)
    )  # , locator=ticker.LogLocator())

    # sets the y axis limits to match the mesh limits
    ax.set_ylim(mesh.r_grid[0], mesh.r_grid[-1])

    plt.colorbar(im, label="Flux")

    plt.show()
