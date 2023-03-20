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
