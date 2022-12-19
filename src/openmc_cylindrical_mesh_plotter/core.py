import matplotlib.pyplot as plt
import numpy as np
import openmc
from matplotlib.colors import LogNorm

def plot_rz_slice(
    tally,
    slice_id=0,
    log = True
):

    mesh = tally.find_filter(openmc.MeshFilter).mesh

    reshaped_tally = tally.mean.reshape(
        (mesh.dimension[2], mesh.dimension[1], mesh.dimension[0])
    )

    tally_aligned = reshaped_tally.transpose(1, 2, 0)
    rotated_slice = np.rot90(tally_aligned[slice_id], 1)

    xmin = mesh.r_grid[0]
    xmax = mesh.r_grid[-1]
    ymin = mesh.z_grid[0]
    ymax = mesh.z_grid[-1]

    if log is True:
        norm = LogNorm()
    else:
        norm = None

    plot = plt.imshow(rotated_slice, extent=[xmin,xmax,ymin,ymax], norm=norm)
    plt.xlabel('R [cm]')
    plt.ylabel('Z [cm]')
    return plt
