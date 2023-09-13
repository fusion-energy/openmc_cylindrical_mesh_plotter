import openmc
import numpy as np
from math import pi
from matplotlib.colors import LogNorm
import pytest
from openmc_cylindrical_mesh_plotter import (
    plot_mesh_tally_rz_slice,
    plot_mesh_tally_phir_slice,
)


@pytest.fixture
def circular_source_simulation():
    mesh = openmc.CylindricalMesh(
        phi_grid=np.linspace(0.0, 2 * pi, 10),
        r_grid=np.linspace(0, 10, 4),
        z_grid=np.linspace(0, 5, 5),
    )
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

    my_geometry = openmc.Geometry([cell])

    my_source = openmc.IndependentSource()

    # the distribution of radius is just a single value
    radius = openmc.stats.Discrete([5], [1])
    # the distribution of source z values is just a single value
    z_values = openmc.stats.Discrete([2.5], [1])
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

    return my_tally_result


@pytest.fixture
def point_source_simulation():
    mesh = openmc.CylindricalMesh(
        phi_grid=np.linspace(0.0, 2 * pi, 10),
        r_grid=np.linspace(0, 10, 4),
        z_grid=np.linspace(0, 5, 5),
    )

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

    my_geometry = openmc.Geometry([cell])

    my_source = openmc.IndependentSource()

    my_source.space = openmc.stats.Point((0, 0.0, 0))

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

    return my_tally_result


def test_rz_slice_of_data_point_simulation_normalization(point_source_simulation):
    tally = point_source_simulation
    mesh = tally.find_filter(openmc.MeshFilter).mesh
    for slice_index in range(0, len(mesh.z_grid) - 1):
        plot_mesh_tally_phir_slice(
            tally=tally,
            slice_index=slice_index,
            # score not specified as it should be automatically found
            # axes "
            # axes_units "
            # value "
            colorbar=True,
            volume_normalization=True,
        )

    outer_surface = openmc.Sphere(r=100, boundary_type="vacuum")
    cell = openmc.Cell(region=-outer_surface)
    for slice_index in range(0, len(mesh.phi_grid) - 1):
        plot_mesh_tally_rz_slice(
            tally=tally,
            slice_index=slice_index,
            score="flux",
            # axes # todo test
            axis_units="m",
            value="std_dev",
            outline=True,
            outline_by="cell",
            geometry=openmc.Geometry([cell]),
            pixels=300,
            colorbar=False,
            volume_normalization=False,
            scaling_factor=10.0,
            colorbar_kwargs={"title": "hi"},
            outline_kwargs={"color": "red"},
            norm=LogNorm(),
        )


def test_phir_slice_of_data_circular_simulation_normalization(
    circular_source_simulation,
):
    tally = circular_source_simulation
    mesh = tally.find_filter(openmc.MeshFilter).mesh

    outer_surface = openmc.Sphere(r=100, boundary_type="vacuum")
    cell = openmc.Cell(region=-outer_surface)
    for slice_index in range(0, len(mesh.z_grid) - 1):
        plot_mesh_tally_rz_slice(
            tally=tally,
            slice_index=slice_index,
            score="flux",
            # axes # todo test
            axis_units="m",
            value="std_dev",
            outline=True,
            outline_by="cell",
            geometry=openmc.Geometry([cell]),
            pixels=300,
            colorbar=False,
            volume_normalization=False,
            scaling_factor=10.0,
            colorbar_kwargs={"title": "hi"},
            outline_kwargs={"color": "red"},
            norm=LogNorm(),
        )
    for slice_index in range(0, len(mesh.phi_grid) - 1):
        plot_mesh_tally_rz_slice(tally=tally, slice_index=slice_index)
