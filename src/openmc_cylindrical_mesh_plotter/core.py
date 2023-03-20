import matplotlib.pyplot as plt
import numpy as np
import openmc
from matplotlib.colors import LogNorm


def plot_rz_ww_slice(
    weight_window: openmc.WeightWindows,
    slice_index: int = 0,
    log: bool = True,
    upper_or_lower: str = 'lower'
):
    if not isinstance(weight_window, openmc.WeightWindows):
        raise TypeError(f'tally argument should be an openmc.WeightWindows object not {type(weight_window)}')

    mesh = weight_window.mesh

    if upper_or_lower == "upper":
        plotted_part_of_weight_window = weight_window.upper_ww_bounds.flatten()
    else:
        plotted_part_of_weight_window = weight_window.lower_ww_bounds.flatten()

    reshaped_tally = plotted_part_of_weight_window.reshape((mesh.dimension[2], mesh.dimension[1], mesh.dimension[0]), order="F")

    tally_aligned = reshaped_tally.transpose(2, 0, 1)

    # TODO add XY top down slice
    # axis_to_slice = 'RZ'
    x_label = "R [cm]"
    y_label = "Z [cm]"

    left = mesh.r_grid[0]
    right = mesh.r_grid[-1]
    bottom = mesh.z_grid[0]
    top = mesh.z_grid[-1]
    # if isinstance(mesh, openmc.CylindricalMesh):

    if log is True:
        norm = LogNorm()
    else:
        norm = None

    image_slice = tally_aligned[slice_index]

    plt.cla()
    plt.clf()

    plt.axes(title=f"Weight window {upper_or_lower}", xlabel=x_label, ylabel=y_label)

    if np.amax(image_slice) == np.amin(image_slice) and norm is not None:
                msg = "slice contains the uniform values, can't be plotted on log scale"
                raise ValueError(msg)    
    else:
        plt.imshow(
            X=image_slice,
            extent=(left, right, bottom, top),
            norm=norm
        )
        plt.colorbar(label=upper_or_lower)

        return plt


def plot_rz_tally_slice(
    tally,
    slice_index=0,
    log = True
):

    if not isinstance(tally, openmc.Tally):
        raise TypeError('tally argument should be an openmc.Tally object')

    mesh = tally.find_filter(openmc.MeshFilter).mesh

    reshaped_tally = tally.mean.reshape(
        (mesh.dimension[2], mesh.dimension[1], mesh.dimension[0])
    )

    tally_aligned = reshaped_tally.transpose(1, 2, 0)
    rotated_slice = np.rot90(tally_aligned[slice_index], 1)

    xmin = mesh.r_grid[0]
    xmax = mesh.r_grid[-1]
    ymin = mesh.z_grid[0]
    ymax = mesh.z_grid[-1]

    if log is True:
        norm = LogNorm()
    else:
        norm = None

    plt.cla()
    plt.clf()

    plot = plt.imshow(rotated_slice, extent=[xmin,xmax,ymin,ymax], norm=norm)
    plt.xlabel('R [cm]')
    plt.ylabel('Z [cm]')
    return plt



def slice_of_data(
    self,
    dataset: np.ndarray,
    # axis='RZ'
    slice_index=0,
    volume_normalization: bool = True,
):

    if volume_normalization:
        dataset = dataset.flatten() / self.volumes.T.reshape(-1, 3).flatten()
    else:
        dataset = dataset.flatten()

    # reshaped_data = dataset.reshape((self.dimension[0], self.dimension[1], self.dimension[2]))
    # reshaped_data = dataset.reshape((self.dimension[0], self.dimension[2], self.dimension[1])) # not bad
    # reshaped_data = dataset.reshape((self.dimension[1], self.dimension[2], self.dimension[0]))
    # reshaped_data = dataset.reshape((self.dimension[1], self.dimension[0], self.dimension[2])) # not this one
    # reshaped_data = dataset.reshape((self.dimension[2], self.dimension[0], self.dimension[1]))
    # reshaped_data = dataset.reshape((self.dimension[2], self.dimension[1], self.dimension[0]))
    # tally_aligned
    data_slice = dataset.T.reshape(-1, 3)

    data_slice = data_slice[slice_index::self.dimension[1]]

    return np.flip(data_slice,axis=0)



def get_mpl_plot_extent(self):
    left = self.r_grid[0]
    right = self.r_grid[-1]
    bottom = self.z_grid[0]
    top = self.z_grid[-1]

    return (left, right, bottom, top)

openmc.CylindricalMesh.slice_of_data = slice_of_data
openmc.mesh.CylindricalMesh.slice_of_data = slice_of_data

openmc.CylindricalMesh.get_mpl_plot_extent = get_mpl_plot_extent
openmc.mesh.CylindricalMesh.get_mpl_plot_extent = get_mpl_plot_extent
