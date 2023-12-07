# this example creates a simple CylindricalMesh tally and performs an openmc
# simulation to populate the tally. Slices of the resulting tally is then
# plotted using the openmc_cylindrical_plotter. In this example both heating
# tallies are combined. This is just a simple tally to show the feature.
# The natural use case is combining neutron and photon dose tallies which must
# be done as separate tallies due to the different energy function filters but
# we often want to combine the neutron and photon dose to get total dose


import openmc
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from openmc_cylindrical_mesh_plotter import plot_mesh_tally_rz_slice

mesh = openmc.CylindricalMesh(
    phi_grid=np.linspace(0.0, 2 * pi, 3),
    r_grid=np.linspace(0, 300, 20),
    z_grid=np.linspace(0, 300, 10),
    origin=(0, 0, 0),
)

neutron_filter = openmc.ParticleFilter("neutron")
photon_filter = openmc.ParticleFilter("photon")

tally1 = openmc.Tally(name="my_neutron_heating_tally")
mesh_filter = openmc.MeshFilter(mesh)
tally1.filters = [mesh_filter, neutron_filter]
tally1.scores = ["heating"]

tally2 = openmc.Tally(name="my_photon_heating_tally")
mesh_filter = openmc.MeshFilter(mesh)
tally2.filters = [mesh_filter, photon_filter]
tally2.scores = ["heating"]

tallies = openmc.Tallies([tally1, tally2])

material = openmc.Material()
material.add_element("Li", 1)
material.set_density("g/cm3", 0.1)
my_materials = openmc.Materials([material])

outer_surface = openmc.Sphere(r=300, boundary_type="vacuum")
cell = openmc.Cell(region=-outer_surface, fill=material)

my_geometry = openmc.Geometry([cell])

source_n = openmc.IndependentSource()
source_n.space = openmc.stats.Point((200, 0, 0))
source_n.angle = openmc.stats.Isotropic()
source_n.energy = openmc.stats.Discrete([0.1e6], [1])
source_n.strength = 1
source_n.particle = "neutron"

source_p = openmc.IndependentSource()
source_p.space = openmc.stats.Point((0, 0, 200))
source_p.angle = openmc.stats.Isotropic()
source_p.energy = openmc.stats.Discrete([10e6], [1])
source_p.strength = 1
source_p.particle = "photon"


my_settings = openmc.Settings()
my_settings.run_mode = "fixed source"
my_settings.batches = 10
my_settings.particles = 100000
my_settings.source = [source_p, source_n]
my_settings.photon_transport = True

model = openmc.model.Model(my_geometry, my_materials, my_settings, tallies)
sp_filename = model.run()

statepoint = openmc.StatePoint(sp_filename)

my_tally1_result = statepoint.get_tally(name="my_neutron_heating_tally")
my_tally2_result = statepoint.get_tally(name="my_photon_heating_tally")

plot = plot_mesh_tally_rz_slice(
    tally=[my_tally1_result, my_tally2_result],
    outline=True,
    geometry=my_geometry,
    # norm=LogNorm(),
    slice_index=1,
    colorbar_kwargs={
        "label": "Neutron and photon heating",
    },
)
plot.figure.savefig(f"rz_point_source_photon_and_neutron_heating.png")
print("written rz_point_source_photon_and_neutron_heating.png")

plot = plot_mesh_tally_rz_slice(
    tally=[my_tally1_result],
    outline=True,
    geometry=my_geometry,
    # norm=LogNorm(),
    slice_index=1,
    colorbar_kwargs={
        "label": "Neutron heating",
    },
)
plot.figure.savefig(f"rz_point_source_neutron_heating.png")
print("written rz_point_source_neutron_heating.png")

plot = plot_mesh_tally_rz_slice(
    tally=[my_tally2_result],
    outline=True,
    geometry=my_geometry,
    # norm=LogNorm(),
    slice_index=1,
    colorbar_kwargs={
        "label": "Photon heating",
    },
)
plot.figure.savefig(f"rz_point_source_photon_heating.png")
print("written rz_point_source_photon_heating.png")
