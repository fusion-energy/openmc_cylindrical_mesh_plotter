import math
from pathlib import Path
from tempfile import TemporaryDirectory
import typing
import openmc
import numpy as np
import openmc
import openmc.checkvalue as cv
import matplotlib.pyplot as plt

from packaging import version

if version.parse(openmc.__version__) > version.parse("0.13.3"):
    pass
else:
    msg = (
        "openmc_regular_mesh_plotter package requires OpenMC version 0.13.4-dev "
        f"or newer. You currently have OpenMC version {openmc.__version__}"
    )
    raise ValueError(msg)

_default_outline_kwargs = {"colors": "black", "linestyles": "solid", "linewidths": 1}

# zero values with logscale produce noise / fuzzy on the time but setting interpolation to none solves this
default_imshow_kwargs = {"interpolation": "none"}


def _check_inputs(
    plot_basis, score, geometry_basis, axis_units, volume_normalization, outline, tally
):
    if plot_basis == "rz":
        cv.check_value("geometry_basis", geometry_basis, ["xz", "yz"])
    else:  # must be 'phir'
        cv.check_value("geometry_basis", geometry_basis, ["xy"])
    cv.check_value("axis_units", axis_units, ["km", "m", "cm", "mm"])
    cv.check_type("volume_normalization", volume_normalization, bool)
    cv.check_type("outline", outline, bool)

    # if tally is multiple tallies
    if isinstance(tally, typing.Sequence):
        mesh_ids = []
        for one_tally in tally:
            mesh = one_tally.find_filter(filter_type=openmc.MeshFilter).mesh
            # TODO check the tallies use the same mesh
            mesh_ids.append(mesh.id)
        if not all(i == mesh_ids[0] for i in mesh_ids):
            raise ValueError(
                f"mesh ids {mesh_ids} are different, please use same mesh when combining tallies"
            )
        # tally is sequence but all meshes are the same
        mesh = one_tally.find_filter(filter_type=openmc.MeshFilter).mesh

    else:
        # tally is single tally
        mesh = tally.find_filter(filter_type=openmc.MeshFilter).mesh

    if not isinstance(mesh, openmc.CylindricalMesh):
        raise NotImplemented(
            f"Only CylindricalMesh are currently supported not {type(mesh)}"
        )

    if mesh.n_dimension != 3:
        msg = "Your mesh has {mesh.n_dimension} dimension and currently only CylindricalMesh with 3 dimensions are supported"
        raise NotImplementedError(msg)

    # if score is not specified and tally has a single score then we know which score to use
    if score is None:
        msg = "score was not specified and there are multiple scores in the tally."
        if isinstance(tally, typing.Sequence):
            for one_tally in tally:
                if len(one_tally.scores) != 1:
                    raise ValueError(msg)
            score = one_tally.scores[0]
        else:
            if len(tally.scores) == 1:
                score = tally.scores[0]
            else:
                raise ValueError(msg)

    return mesh, score


def _get_tally_data(
    plot_basis,
    scaling_factor,
    mesh,
    tally,
    value,
    volume_normalization,
    score,
    slice_index,
):
    tally_slice = tally.get_slice(scores=[score])

    tally_slice = tally.get_slice(scores=[score])

    tally_data = tally_slice.get_reshaped_data(expand_dims=True, value=value).squeeze()

    if slice_index is None:
        dim_index = {"rz": 1, "phir": 2}[plot_basis]
        # get the middle phi or z value
        slice_index = int(tally_data.shape[dim_index] / 2)  # index 1 is the phi value

    if len(tally_data.shape) == 3:
        if plot_basis == "rz":
            data = tally_data[:, slice_index, :]
        else:  # phir
            data = tally_data[:, :, slice_index]
    elif len(tally_data.shape) == 2:
        data = tally_data[:, :]
    else:
        raise NotImplementedError("Mesh is not 3d or 2d, can't plot")

    if volume_normalization:
        if len(tally_data.shape) == 3:
            if plot_basis == "rz":
                slice_volumes = mesh.volumes[:, slice_index, :].squeeze()
            else:  # phir
                slice_volumes = mesh.volumes[:, :, slice_index].squeeze()
        elif len(tally_data.shape) == 2:
            slice_volumes = mesh.volumes[:, :].squeeze()
        data = data / slice_volumes

    if scaling_factor:
        data = data * scaling_factor

    if plot_basis == "rz":
        data = np.rot90(data, 1)
    return data


def plot_mesh_tally_rz_slice(
    tally: typing.Union["openmc.Tally", typing.Sequence["openmc.Tally"]],
    slice_index: typing.Optional[int] = None,
    score: typing.Optional[str] = None,
    axes: typing.Optional[str] = None,
    axis_units: str = "cm",
    value: str = "mean",
    outline: bool = False,
    outline_by: str = "cell",
    geometry: typing.Optional["openmc.Geometry"] = None,
    geometry_basis: str = "xz",
    pixels: int = 40000,
    colorbar: bool = True,
    volume_normalization: bool = True,
    mirror: bool = False,
    scaling_factor: typing.Optional[float] = None,
    colorbar_kwargs: dict = {},
    outline_kwargs: dict = _default_outline_kwargs,
    **kwargs,
) -> "matplotlib.image.AxesImage":
    """Display a slice plot of the mesh tally score.
    Parameters
    ----------
    tally : openmc.Tally
        The openmc tally/tallies to plot. Tally must contain a MeshFilter that
        uses a CylindricalMesh. If a sequence of multiple tallies are provided
        the score will be added.
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
    geometry_basis : str
        The axis to use for the geometry slice
    pixels : int
        This sets the total number of pixels in the plot and the number of
        pixels in each basis direction is calculated from this total and
        the image aspect ratio.
    colorbar : bool
        Whether or not to add a colorbar to the plot.
    volume_normalization : bool, optional
        Whether or not to normalize the data by the volume of the mesh elements.
    mirror : bool, optional
        Whether to reflect the plot in the z axis to include negative r values.
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

    mesh, score = _check_inputs(
        "rz", score, geometry_basis, axis_units, volume_normalization, outline, tally
    )

    xlabel, ylabel = f"r [{axis_units}]", f"z [{axis_units}]"
    axis_scaling_factor = {"km": 0.00001, "m": 0.01, "cm": 1, "mm": 10}[axis_units]

    if mesh.origin[0] != 0.0 or mesh.origin[1] != 0.0:
        raise ValueError(
            "Plotter only works for cylindrical meshes with x,y origins of 0,0"
        )

    extent = [
        mesh.r_grid[0],
        mesh.r_grid[-1],
        mesh.origin[2] + mesh.z_grid[0],
        mesh.origin[2] + mesh.z_grid[-1],
    ]

    x_min, x_max, y_min, y_max = [i * axis_scaling_factor for i in extent]

    if axes is None:
        fig, axes = plt.subplots()
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)

    default_imshow_kwargs.update(kwargs)

    if isinstance(tally, typing.Sequence):
        for counter, one_tally in enumerate(tally):
            new_data = _get_tally_data(
                "rz",
                scaling_factor,
                mesh,
                one_tally,
                value,
                volume_normalization,
                score,
                slice_index,
            )
            if counter == 0:
                data = np.zeros(shape=new_data.shape)
            data = data + new_data
    else:  # single tally
        data = _get_tally_data(
            "rz",
            scaling_factor,
            mesh,
            tally,
            value,
            volume_normalization,
            score,
            slice_index,
        )

    if mirror:
        data_reflected = np.fliplr(data)
        data = np.concatenate((data_reflected, data), axis=1)
        x_min = x_max * -1

    im = axes.imshow(data, extent=(x_min, x_max, y_min, y_max), **default_imshow_kwargs)

    if colorbar:
        fig.colorbar(im, **colorbar_kwargs)

    if outline and geometry is not None:
        import matplotlib.image as mpimg

        model = openmc.Model()
        model.geometry = geometry
        plot = openmc.Plot()

        width_x = abs(extent[1] - extent[0])
        width_y = abs(extent[1] - extent[0])  # same
        width_z = abs(extent[3] - extent[2])

        x_center = extent[0] + (width_x / 2)
        y_center = extent[0] + (width_y / 2)
        z_center = extent[2] + width_z * 0.5

        if geometry_basis == "xz":
            plot.origin = (x_center, 0, z_center)
            plot.width = (width_x, width_z)
            aspect_ratio = width_x / width_z
        else:  # geometry_basis='yz'
            plot.origin = (0, y_center, z_center)
            plot.width = (width_y, width_z)
            aspect_ratio = width_y / width_z

        pixels_y = math.sqrt(pixels / aspect_ratio)
        pixels = (int(pixels / pixels_y), int(pixels_y))
        plot.pixels = pixels
        plot.basis = geometry_basis
        plot.color_by = outline_by
        model.plots.append(plot)

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

        if mirror:
            image_value_reflected = np.fliplr(image_value)
            image_value = np.concatenate((image_value_reflected, image_value), axis=1)

        # Plot geometry image
        axes.contour(
            image_value,
            origin="upper",
            levels=np.unique(image_value),
            extent=(x_min, x_max, y_min, y_max),
            **outline_kwargs,
        )

    return axes


def plot_mesh_tally_phir_slice(
    tally: "openmc.Tally",
    slice_index: typing.Optional[int] = None,
    score: typing.Optional[str] = None,
    axes: typing.Optional[str] = None,
    axis_units: str = "cm",
    value: str = "mean",
    outline: bool = False,
    outline_by: str = "cell",
    geometry: typing.Optional["openmc.Geometry"] = None,
    pixels: int = 40000,
    colorbar: bool = True,
    volume_normalization: bool = True,
    scaling_factor: typing.Optional[float] = None,
    colorbar_kwargs: dict = {},
    outline_kwargs: dict = _default_outline_kwargs,
    **kwargs,
) -> "matplotlib.image.AxesImage":
    """Display a slice plot of the mesh tally score.
    Parameters
    ----------
    tally : openmc.Tally
        The openmc tally to plot. Tally must contain a MeshFilter that uses a CylindricalMesh.
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
    geometry_basis: str = "xy"  # TODO add geometry outline plot to phiR plotting

    mesh, score = _check_inputs(
        "phir", score, geometry_basis, axis_units, volume_normalization, outline, tally
    )

    xlabel, ylabel = f"r [{axis_units}]", f"z [{axis_units}]"
    axis_scaling_factor = {"km": 0.00001, "m": 0.01, "cm": 1, "mm": 10}[axis_units]

    extent = [mesh.r_grid[0], mesh.r_grid[-1], mesh.z_grid[0], mesh.z_grid[-1]]

    x_min, x_max, y_min, y_max = [i * axis_scaling_factor for i in extent]

    if axes is None:
        fig, axes = plt.subplots(subplot_kw=dict(projection="polar"))
        # axes.set_xlabel(xlabel)
        # axes.set_ylabel(ylabel)

    theta = np.linspace(mesh.phi_grid[0], mesh.phi_grid[-1], len(mesh.phi_grid) - 1)
    r = np.linspace(mesh.r_grid[0], mesh.r_grid[-1], len(mesh.r_grid) - 1)

    # default_imshow_kwargs.update(kwargs)

    if isinstance(tally, typing.Sequence):
        for counter, one_tally in enumerate(tally):
            new_data = _get_tally_data(
                "phir",
                scaling_factor,
                mesh,
                one_tally,
                value,
                volume_normalization,
                score,
                slice_index,
            )
            if counter == 0:
                data = np.zeros(shape=new_data.shape)
            data = data + new_data
    else:  # single tally
        data = _get_tally_data(
            "phir",
            scaling_factor,
            mesh,
            tally,
            value,
            volume_normalization,
            score,
            slice_index,
        )

    im = axes.contourf(theta[:], r[:], data, extent=(0, 100, 0, 50), **kwargs)

    if colorbar:
        fig.colorbar(im, **colorbar_kwargs)

    # todo see if moving from a contourf to imshow can work
    # https://stackoverflow.com/questions/54209640/imshow-in-polar-coordinates

    # if outline and geometry is not None:
    #     import matplotlib.image as mpimg

    #     # code to make sure geometry outline is in the middle of the mesh voxel
    #     # two of the three dimensions are just in the center of the mesh
    #     # but the slice can move one axis off the center so this needs calculating
    #     x1, y1, z1 = mesh.upper_right
    #     x_origin, y_origin, z_origin = mesh.origin

    #     width_x = x_origin + x1
    #     width_y = y_origin + y1
    #     width_z = z_origin + z1

    #     model = openmc.Model()
    #     model.geometry = geometry
    #     plot = openmc.Plot()

    #     if geometry_basis == "xz":
    #         plot.origin = (0.5 * width_x, 0.0, 0.5 * width_z)
    #         plot.width = (width_x, width_z)
    #         aspect_ratio = width_x / width_z
    #     else:  # geometry_basis='xz'
    #         plot.origin = (0, 0.5 * width_y, 0.5 * width_z)
    #         plot.width = (width_y, width_z)
    #         aspect_ratio = width_y / width_z

    #     pixels_y = math.sqrt(pixels / aspect_ratio)
    #     pixels = (int(pixels / pixels_y), int(pixels_y))
    #     plot.pixels = pixels
    #     plot.basis = geometry_basis
    #     plot.color_by = outline_by
    #     model.plots.append(plot)

    #     with TemporaryDirectory() as tmpdir:
    #         # Run OpenMC in geometry plotting mode
    #         model.plot_geometry(False, cwd=tmpdir)

    #         # Read image from file
    #         img_path = Path(tmpdir) / f"plot_{plot.id}.png"
    #         if not img_path.is_file():
    #             img_path = img_path.with_suffix(".ppm")
    #         img = mpimg.imread(str(img_path))

    #     # Combine R, G, B values into a single int
    #     rgb = (img * 256).astype(int)
    #     image_value = (rgb[..., 0] << 16) + (rgb[..., 1] << 8) + (rgb[..., 2])

    #     # Plot geometry image
    #     axes.contour(
    #         image_value,
    #         origin="upper",
    #         levels=np.unique(image_value),
    #         extent=(x_min, x_max, y_min, y_max),
    #         **outline_kwargs,
    #     )

    return axes
