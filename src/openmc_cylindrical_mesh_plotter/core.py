import matplotlib.pyplot as plt
import numpy as np
import openmc


def slice_of_data(
    self,
    dataset: np.ndarray,
    # axis='RZ' todo add more axis including top down view
    slice_index=0,
    volume_normalization: bool = True,
):
    if volume_normalization:
        dataset = dataset.flatten() / self.volumes.T.reshape(-1, 3).flatten()
    else:
        dataset = dataset.flatten()

    data_slice = dataset.T.reshape(-1, 3)

    data_slice = data_slice[slice_index :: self.dimension[1]]

    return np.flip(data_slice, axis=0)


def get_mpl_plot_extent(self):
    """Returns the (x_min, x_max, y_min, y_max) of the mesh based on the
    r_grid and z_grid."""
    left = self.r_grid[0]
    right = self.r_grid[-1]
    bottom = self.z_grid[0]
    top = self.z_grid[-1]

    return (left, right, bottom, top)


def get_axis_labels(self):
    """Returns two axis label values for the x and y value. Takes
    view_direction into account."""

    xlabel = "R [cm]"
    ylabel = "Z [cm]"

    return xlabel, ylabel


openmc.CylindricalMesh.slice_of_data = slice_of_data
openmc.mesh.CylindricalMesh.slice_of_data = slice_of_data

openmc.CylindricalMesh.get_axis_labels = get_axis_labels
openmc.mesh.CylindricalMesh.get_axis_labels = get_axis_labels

openmc.CylindricalMesh.get_mpl_plot_extent = get_mpl_plot_extent
openmc.mesh.CylindricalMesh.get_mpl_plot_extent = get_mpl_plot_extent
