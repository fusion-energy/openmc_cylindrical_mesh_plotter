# this example creates a simple CylindricalMesh tally and performs an openmc
# simulation to populate the tally. Slices of the resulting tally is then
# plotted using the openmc_cylindrical_plotter in the Phi R axis.

import openmc
import numpy as np
from math import pi
from openmc_cylindrical_mesh_plotter import plot_mesh_tally_rz_slice

material = openmc.Material()
material.add_nuclide("Fe56", 1)
material.set_density("g/cm3", 0.1)
my_materials = openmc.Materials([material])

inner_surface = openmc.Sphere(r=20)
outer_surface = openmc.model.RectangularParallelepiped(
    0, 100, 0, 100, 0, 100, boundary_type="vacuum"
)
cell_inner = openmc.Cell(region=-outer_surface & -inner_surface, fill=material)
cell_outer = openmc.Cell(region=-outer_surface & +inner_surface, fill=material)

my_geometry = openmc.Geometry([cell_inner, cell_outer])

my_source = openmc.IndependentSource()
# this makes a point source instead with
my_source.space = openmc.stats.Point(my_geometry.bounding_box.center)
# sets the direction to isotropic
# my_source.angle = openmc.stats.Isotropic()


my_settings = openmc.Settings()
# my_settings.inactive = 0
my_settings.run_mode = "fixed source"
my_settings.batches = 10
my_settings.particles = 100000
my_settings.source = my_source

mesh = openmc.CylindricalMesh.from_domain(domain=my_geometry)

tally = openmc.Tally(name="my_tally")
mesh_filter = openmc.MeshFilter(mesh)
tally.filters.append(mesh_filter)
tally.scores.append("flux")
tallies = openmc.Tallies([tally])
model = openmc.model.Model(my_geometry, my_materials, my_settings, tallies)

sp_filename = model.run()

statepoint = openmc.StatePoint(sp_filename)

my_tally_result = statepoint.get_tally(name="my_tally")

for slice_index in range(1, len(mesh.z_grid)):
    plot = plot_mesh_tally_rz_slice(
        tally=my_tally_result, outline=True, geometry=my_geometry
    )

    plot.figure.savefig(f"phir_{slice_index}.png")
