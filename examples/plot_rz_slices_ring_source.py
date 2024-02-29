# this example creates a simple CylindricalMesh tally and performs an openmc
# simulation to populate the tally. Slices of the resulting tally is then
# plotted using the openmc_cylindrical_plotter in the R Z axis

import openmc
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from openmc_cylindrical_mesh_plotter import plot_mesh_tally_rz_slice
from matplotlib.colors import LogNorm

mesh = openmc.CylindricalMesh(
    phi_grid=np.linspace(0.0, 2 * pi, 4),
    r_grid=np.linspace(0, 100, 50),
    z_grid=np.linspace(-100, 50, 50),
)
mesh_filter = openmc.MeshFilter(mesh)

tally = openmc.Tally(name="my_tally")
tally.filters = [mesh_filter]
tally.scores = ["flux"]
tallies = openmc.Tallies([tally])

surf1 = openmc.model.RectangularParallelepiped(
    -100, 100, -100, 100, -100, 50, boundary_type="vacuum"
)
surf2 = openmc.model.RectangularParallelepiped(-95, 95, -95, 95, -95, 45)
surf3 = openmc.Sphere(r=40)
cell1 = openmc.Cell(region=-surf3)
cell2 = openmc.Cell(region=+surf2 & -surf1)
cell3 = openmc.Cell(region=+surf3 & -surf2)

material = openmc.Material()
material.add_nuclide("Fe56", 1)
material.set_density("g/cm3", 0.1)
my_materials = openmc.Materials([material])

my_geometry = openmc.Geometry([cell1, cell2, cell3])

my_source = openmc.Source()

# the distribution of radius is just a single value
radius = openmc.stats.Discrete([5], [1])
# the distribution of source z values is just a single value
z_values = openmc.stats.Discrete([4], [1])
# the distribution of source azimuthal angles values is a uniform distribution between 0 and 2 Pi
angle = openmc.stats.Uniform(a=0.0, b=2 * 3.14159265359)
# this makes the ring source using the three distributions and a radius
# could do a point source instead with my_source.space = openmc.stats.Point((5,5., 5))
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

for slice_index in range(1, len(mesh.phi_grid) - 1):
    plot = plot_mesh_tally_rz_slice(
        tally=my_tally_result,
        geometry=my_geometry,
        outline=True,
        norm=LogNorm(),
        slice_index=slice_index,
        mirror=True,
    )
    plot.figure.savefig(f"rz_ring_source_reflected_{slice_index}.png")

    plot = plot_mesh_tally_rz_slice(
        tally=my_tally_result,
        geometry=my_geometry,
        outline=True,
        norm=LogNorm(),
        slice_index=slice_index,
    )
    plot.figure.savefig(f"rz_ring_source_{slice_index}.png")
