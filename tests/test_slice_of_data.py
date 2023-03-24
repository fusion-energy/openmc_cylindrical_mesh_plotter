import openmc
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import openmc_cylindrical_mesh_plotter  # adds slice_of_data method to CylindricalMesh
import pytest


mesh = openmc.CylindricalMesh()
mesh.phi_grid = np.linspace(0.0, 2 * pi, 10)
mesh.r_grid = np.linspace(0, 10, 4)
mesh.z_grid = np.linspace(0, 5, 5)


@pytest.fixture
def circular_source_simulation():
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

    return my_tally_result.mean.flatten()


@pytest.fixture
def point_source_simulation():
    mesh = openmc.CylindricalMesh()
    mesh.phi_grid = np.linspace(0.0, 2 * pi, 10)
    mesh.r_grid = np.linspace(0, 10, 4)
    mesh.z_grid = np.linspace(0, 5, 5)

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

    return my_tally_result.mean.flatten()


@pytest.fixture
def flat_data():
    flat_data = [1] * len(mesh.phi_grid) * len(mesh.r_grid) * len(mesh.z_grid)
    return np.array(flat_data)


def test_get_mpl_plot_extent():
    pass
    # todo get extern for both plots polar and imshow


def test_get_axis_labels():
    # todo get labels for both plots polar and imshow
    pass


def test_rz_slice_of_data_flat_data_normalized(flat_data):
    for slice_index in range(len(mesh.phi_grid) - 1):
        data = mesh.slice_of_data(
            dataset=flat_data,
            axis="RZ",
            slice_index=slice_index,
            volume_normalization=True,
        )

        assert data.shape == (4, 3)

    # TODO check the data values are smaller on the right than the left
    # source is in the middle, voxel volumes are smaller in the center


def test_rz_slice_of_data_flat_data_unnormalized(flat_data):
    for slice_index in range(len(mesh.phi_grid) - 1):
        data = mesh.slice_of_data(
            dataset=flat_data,
            axis="RZ",
            slice_index=slice_index,
            volume_normalization=False,
        )

        assert data.shape == (4, 3)

    # TODO check the data values equal everywhere


def test_phir_slice_of_data_flat_data_normalized(flat_data):
    for slice_index in range(len(mesh.phi_grid) - 1):
        data = mesh.slice_of_data(
            dataset=flat_data,
            axis="PhiR",
            slice_index=slice_index,
            volume_normalization=True,
        )

        assert data.shape == (4, 3)

    # TODO check the data values


def test_phir_slice_of_data_flat_data_unnormalized(flat_data):
    for slice_index in range(len(mesh.phi_grid) - 1):
        data = mesh.slice_of_data(
            dataset=flat_data,
            axis="RZ",
            slice_index=slice_index,
            volume_normalization=False,
        )

        assert data.shape == (4, 3)

    # TODO check the data values equal everywhere


def test_rz_slice_of_data_point_simulation_normalization(point_source_simulation):
    for slice_index in range(len(mesh.phi_grid) - 1):
        data = mesh.slice_of_data(
            dataset=point_source_simulation,
            axis="RZ",
            slice_index=slice_index,
            volume_normalization=True,
        )

        assert data.shape == (4, 3)

    # TODO test


def test_phir_slice_of_data_circular_simulation_normalization(
    circular_source_simulation,
):
    for slice_index in range(len(mesh.phi_grid) - 1):
        data = mesh.slice_of_data(
            dataset=circular_source_simulation,
            axis="PhiR",
            slice_index=slice_index,
            volume_normalization=True,
        )

        assert data.shape == (4, 3)

    # TODO test


def test_rz_slice_of_data_point_simulation_unnormalization(point_source_simulation):
    for slice_index in range(len(mesh.phi_grid) - 1):
        data = mesh.slice_of_data(
            dataset=point_source_simulation,
            axis="RZ",
            slice_index=slice_index,
            volume_normalization=False,
        )

        assert data.shape == (4, 3)

    # TODO test


def test_phir_slice_of_data_circular_simulation_unnormalization(
    circular_source_simulation,
):
    for slice_index in range(len(mesh.phi_grid) - 1):
        theta, r, values = mesh.slice_of_data(
            dataset=circular_source_simulation,
            axis="PhiR",
            slice_index=slice_index,
            volume_normalization=False,
        )
        theta.shape == (4, 3)
        r.shape == (4, 3)
        values.shape == (4, 3)

    # TODO test
