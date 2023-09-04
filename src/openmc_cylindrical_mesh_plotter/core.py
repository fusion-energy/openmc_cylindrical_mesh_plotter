import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
import openmc
import numpy as np
import openmc
import openmc.checkvalue as cv
import matplotlib.pyplot as plt

from packaging import version

if version.parse(openmc.__version__) < version.parse("0.13.3"):
    msg = (
        "openmc_regular_mesh_plotter package requires OpenMC version 0.13.4 "
        f"or newer. You currently have OpenMC version {openmc.__version__}"
    )
    raise ValueError(msg)

_BASES = ["rz", "phiz"]

_default_outline_kwargs = {"colors": "black", "linestyles": "solid", "linewidths": 1}


def plot_mesh_tally(
    tally: "openmc.Tally",
    basis: str = "rz",
    slice_index: Optional[int] = None,
    score: Optional[str] = None,
    axes: Optional[str] = None,
    axis_units: str = "cm",
    value: str = "mean",
    outline: bool = False,
    outline_by: str = "cell",
    geometry: Optional["openmc.Geometry"] = None,
    geometry_basis: 'xy',
    pixels: int = 40000,
    colorbar: bool = True,
    volume_normalization: bool = True,
    scaling_factor: Optional[float] = None,
    colorbar_kwargs: dict = {},
    outline_kwargs: dict = _default_outline_kwargs,
    **kwargs,
) -> "matplotlib.image.AxesImage":
    """Display a slice plot of the mesh tally score.
    Parameters
    ----------
    tally : openmc.Tally
        The openmc tally to plot. Tally must contain a MeshFilter that uses a CylindricalMesh.
    basis : {'rz', 'phiz'}
        The basis directions for the plot
    slice_index : int
        The mesh index to plot
    score : str
        Score to plot, e.g. 'flux'
    axes : matplotlib.Axes
        Axes to draw to
    axis_units : {'km', 'm', 'cm', 'mm'}
        Units used on the plot axis
    value : str
        A string for the type of value to return  - 'mean' (default),
        'std_dev', 'rel_err', 'sum', or 'sum_sq' are accepted
    outline : True
        If set then an outline will be added to the plot. The outline can be
        by cell or by material.
    outline_by : {'cell', 'material'}
        Indicate whether the plot should be colored by cell or by material
    geometry : openmc.Geometry
        The geometry to use for the outline.
    pixels : int
        This sets the total number of pixels in the plot and the number of
        pixels in each basis direction is calculated from this total and
        the image aspect ratio.
    colorbar : bool
        Whether or not to add a colorbar to the plot.
    volume_normalization : bool, optional
        Whether or not to normalize the data by the volume of the mesh elements.
    scaling_factor : float
        A optional multiplier to apply to the tally data prior to ploting.
    colorbar_kwargs : dict
        Keyword arguments passed to :func:`matplotlib.colorbar.Colorbar`.
    outline_kwargs : dict
        Keyword arguments passed to :func:`matplotlib.pyplot.contour`.
    **kwargs
        Keyword arguments passed to :func:`matplotlib.pyplot.imshow`
    Returns
    -------
    matplotlib.image.AxesImage
        Resulting image
    """

    cv.check_value("basis", basis, _BASES)
    cv.check_value("axis_units", axis_units, ["km", "m", "cm", "mm"])
    cv.check_type("volume_normalization", volume_normalization, bool)
    cv.check_type("outline", outline, bool)

    mesh = tally.find_filter(filter_type=openmc.MeshFilter).mesh
    if not isinstance(mesh, openmc.CylindricalMesh):
        raise NotImplemented(
            f"Only CylindricalMesh are currently supported not {type(mesh)}"
        )
    # if mesh.n_dimension != 3:
    #     msg = "Your mesh has {mesh.n_dimension} dimension and currently only CylindricalMesh with 3 dimensions are supported"
    #     raise NotImplementedError(msg)

    # if score is not specified and tally has a single score then we know which score to use
    if score is None:
        if len(tally.scores) == 1:
            score = tally.scores[0]
        else:
            msg = "score was not specified and there are multiple scores in the tally."
            raise ValueError(msg)

    tally_slice = tally.get_slice(scores=[score])

    tally_data = tally_slice.get_reshaped_data(expand_dims=True, value=value).squeeze()

    print('tally_data.shape', tally_data.shape)

    if slice_index is None:
        basis_to_index = {"rz": 1, "phir": 0}[basis]  # todo check phir
        slice_index = int(tally_data.shape[basis_to_index] / 2)

    if basis == "rz":
        data = tally_data[:, slice_index, :]
        # data = np.rot90(slice_data, 1)
        xlabel, ylabel = f"r [{axis_units}]", f"z [{axis_units}]"
    else:  # basis == 'phir'
        # todo
        pass

    print('data.shape', data.shape)
    if volume_normalization:
        slice_volumes = mesh.volumes[:, slice_index, :]
        data = data / slice_volumes

    if scaling_factor:
        data = data * scaling_factor

    if basis == "rz":
        data = np.rot90(data, 1)

    axis_scaling_factor = {"km": 0.00001, "m": 0.01, "cm": 1, "mm": 10}[axis_units]

    if basis == "rz":
        extent = [mesh.r_grid[0], mesh.r_grid[-1], mesh.z_grid[0], mesh.z_grid[-1]]
    else:
        raise NotImplementedError("todo extent for phi basis plot")

    x_min, x_max, y_min, y_max = [i * axis_scaling_factor for i in extent]

    if axes is None:
        fig, axes = plt.subplots()
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)

    im = axes.imshow(data, extent=(x_min, x_max, y_min, y_max), **kwargs)

    if colorbar:
        fig.colorbar(im, **colorbar_kwargs)

    if outline and geometry is not None:
        import matplotlib.image as mpimg

        # code to make sure geometry outline is in the middle of the mesh voxel
        # two of the three dimensions are just in the center of the mesh
        # but the slice can move one axis off the center so this needs calculating
        x0, y0, z0 = mesh.lower_left
        x1, y1, z1 = mesh.upper_right
        xo, yo, zo = mesh.origin
        nx, ny, nz = mesh.dimension

        xc = (xo + x1) / 2
        yc = (yo + y1) / 2
        zc = mesh.bounding_box.center[2]
        print(' mesh.bounding_box',  mesh.bounding_box)

        width_x = (xo+x1)
        width_y = (yo+y1)
        width_z = (zo+z1)

        center_of_mesh = mesh.bounding_box.center
        if basis == "rz":
            center_of_mesh_slice = [xc, 0., zc]
        if basis == "phir":
            raise NotImplementedError("outline with phir basis is not yet supported, contributions are welcome :-)")

        geometry_basis='xz' # TODO allow rz to result in xy or yz

        print('mesh.bounding_box.center', mesh.bounding_box.center)
        print('center_of_mesh_slice', center_of_mesh_slice)
        print('mesh.origin', mesh.origin)
# mesh.bounding_box.center [0. 0. 0.]
# center_of_mesh_slice [1250.0, 1250.0, 0.0]
# mesh.origin [0. 0. 0.]

        model = openmc.Model()
        model.geometry = geometry
        plot = openmc.Plot()
        plot.origin = (0.5*width_x,0.,0.5*width_z)#(1250, 0 , 1000)#center_of_mesh_slice
        # bb_width = mesh.bounding_box.extent[basis]
        plot.width = (width_x, width_z) # tod might use width y if yz basis used
        aspect_ratio = width_x / width_y
        pixels_y = math.sqrt(pixels / aspect_ratio)
        pixels = (int(pixels / pixels_y), int(pixels_y))
        plot.pixels = pixels
        plot.basis = geometry_basis
        plot.color_by = outline_by
        model.plots.append(plot)
        print(plot)

        with TemporaryDirectory() as tmpdir:
            # Run OpenMC in geometry plotting mode
            model.plot_geometry(False, cwd=tmpdir)

            # Read image from file
            img_path = Path(tmpdir) / f"plot_{plot.id}.png"
            if not img_path.is_file():
                img_path = img_path.with_suffix(".ppm")
            img = mpimg.imread(str(img_path))

        # Combine R, G, B values into a single int
        rgb = (img * 256).astype(int)
        image_value = (rgb[..., 0] << 16) + (rgb[..., 1] << 8) + (rgb[..., 2])

        if geometry_basis == "xz":
            pass
            # image_value = np.rot90(image_value, 2)
        elif geometry_basis == "yz":
            image_value = np.rot90(image_value, 2)
        else:  # geometry_basis == 'xy'
            image_value = np.rot90(image_value, 2)

        # Plot image and return the axes
        axes.contour(
            image_value,
            origin="upper",
            levels=np.unique(image_value),
            extent=(x_min, x_max, y_min, y_max),
            **outline_kwargs,
        )

    return axes
