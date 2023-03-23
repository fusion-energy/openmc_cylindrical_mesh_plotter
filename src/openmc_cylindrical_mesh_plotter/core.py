import matplotlib.pyplot as plt
import numpy as np
import openmc


def slice_of_data(
    self,
    dataset: np.ndarray,
    axis: str,
    slice_index=0,
    volume_normalization: bool = True,
):
    if axis == 'R-Z':
        return slice_of_rz_data(
            self,
            dataset=dataset,
            slice_index=slice_index,
            volume_normalization=volume_normalization
        )
    elif axis == 'Phi-R':
        return slice_of_phir_data(
            self,
            dataset=dataset,
            slice_index=slice_index,
            volume_normalization=volume_normalization
        )
    else:
        raise ValueError(f'axis must be either "R-Z" or "Phi-R", not {axis}')


def slice_of_phir_data(
    self,
    dataset: np.ndarray,
    slice_index=0,
    volume_normalization: bool = True,
):
    actual = np.linspace(self.phi_grid[0], self.phi_grid[-1], self.dimension[1])
    expected = np.linspace(self.r_grid[0], self.r_grid[-1], self.dimension[0])

    r, theta = np.meshgrid(expected, actual)

    lower_index = int(slice_index * (len(self.phi_grid) - 1))
    upper_index = int((slice_index + 1) * (len(self.phi_grid) - 1))

    # both order A and C appear to work
    # values=dataset.flatten().reshape(-1,len(self.r_grid)-1,order='C')[:len(self.phi_grid)-1]
    values = dataset.flatten().reshape(-1, len(self.r_grid) - 1, order="A")[
        lower_index:upper_index
    ]

    return theta, r, values

def slice_of_rz_data(
    self,
    dataset: np.ndarray,
    slice_index=0,
    volume_normalization: bool = True,
):

    lower_index = int(slice_index * (len(self.r_grid) - 1))
    upper_index = int((slice_index + 1) * (len(self.r_grid) - 1))

    if volume_normalization:
        data_slice = dataset.flatten().reshape(-1, len(self.z_grid) - 1, order="F")[
            lower_index:upper_index
        ]
        data_slice = data_slice / self.volumes[:, 1, :]

        return np.rot90(data_slice)

    data_slice = dataset.flatten().reshape(-1, len(self.z_grid) - 1, order="F")[
        lower_index:upper_index
    ]

    return np.rot90(data_slice)


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
