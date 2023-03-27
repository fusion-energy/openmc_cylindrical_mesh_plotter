import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openmc


def slice_of_data(
    self,
    dataset: np.ndarray,
    view_direction: str,
    slice_index=0,
    volume_normalization: bool = True,
):
    if view_direction == "RZ":
        return slice_of_rz_data(
            self,
            dataset=dataset,
            slice_index=slice_index,
            volume_normalization=volume_normalization,
        )
    elif view_direction == "PhiR":
        return slice_of_phir_data(
            self,
            dataset=dataset,
            slice_index=slice_index,
            volume_normalization=volume_normalization,
        )
    else:
        raise ValueError(f'view_direction must be either "RZ" or "PhiR", not {view_direction}')


def slice_of_phir_data(
    self,
    dataset: np.ndarray,
    slice_index=0,
    volume_normalization: bool = True,
):
    actual = np.linspace(self.phi_grid[0], self.phi_grid[-1], self.dimension[1])
    expected = np.linspace(self.r_grid[0], self.r_grid[-1], self.dimension[0])

    rad, theta = np.meshgrid(expected, actual)

    # old method which suffered from errors reshaping but avoided using pandas
    # lower_index = int(slice_index * (len(self.phi_grid) - 1))
    # upper_index = int((slice_index + 1) * (len(self.phi_grid) - 1))

    # print(dataset.flatten())
    # # both order A and C appear to work
    # # values=dataset.flatten().reshape(-1,len(self.r_grid)-1,order='C')[:len(self.phi_grid)-1]
    # values = dataset.flatten().reshape(-1, len(self.r_grid) - 1, order="A")[
    #     lower_index:upper_index
    # ]

    # return theta, r, values
    indices = list(self.indices)

    r = [entry[0] for entry in indices]
    phi = [entry[1] for entry in indices]
    z = [entry[2] for entry in indices]

    df = pd.DataFrame(
        {
            "r": r,
            "phi": phi,
            "z": z,
            "value": dataset,
            "volume": self.volumes.flatten(),
            "normalised_value": dataset / self.volumes.flatten(),
        }
    )
    df_slice = df[df["z"] == slice_index]

    if volume_normalization:
        shaped_slice = (
            df_slice["normalised_value"].to_numpy().reshape(-1, len(self.r_grid) - 1)
        )
    else:
        shaped_slice = df_slice["value"].to_numpy().reshape(-1, len(self.r_grid) - 1)

    return theta, rad, shaped_slice


def slice_of_rz_data(
    self,
    dataset: np.ndarray,
    slice_index=0,
    volume_normalization: bool = True,
):
    # old method which suffered from errors reshaping but avoided using pandas
    # lower_index = int(slice_index * (len(self.r_grid) - 1))
    # upper_index = int((slice_index + 1) * (len(self.r_grid) - 1))

    # if volume_normalization:
    #     data_slice = dataset.flatten().reshape(-1, len(self.z_grid) - 1, order="F")[
    #         lower_index:upper_index
    #     ]
    #     data_slice = data_slice / self.volumes[:, 1, :]

    #     return np.rot90(data_slice)

    # data_slice = dataset.flatten().reshape(-1, len(self.z_grid) - 1, order="F")[
    #     lower_index:upper_index
    # ]

    # return np.rot90(data_slice)

    indices = list(self.indices)

    r = [entry[0] for entry in indices]
    phi = [entry[1] for entry in indices]
    z = [entry[2] for entry in indices]

    df = pd.DataFrame(
        {
            "r": r,
            "phi": phi,
            "z": z,
            "value": dataset,
            "volume": self.volumes.flatten(),
            "normalised_value": dataset / self.volumes.flatten(),
        }
    )
    df_slice = df[df["phi"] == slice_index]

    if volume_normalization:
        shaped_slice = (
            df_slice["normalised_value"].to_numpy().reshape(-1, len(self.r_grid) - 1)
        )
    else:
        shaped_slice = df_slice["value"].to_numpy().reshape(-1, len(self.r_grid) - 1)

    return np.flipud(shaped_slice)


def get_mpl_plot_extent(self, view_direction='RZ'):
    """Returns the (x_min, x_max, y_min, y_max) of the mesh based on the
    r_grid and z_grid."""
    if view_direction == 'RZ':
        left = self.r_grid[0]
        right = self.r_grid[-1]
        bottom = self.z_grid[0]
        top = self.z_grid[-1]
        return (left, right, bottom, top)
    elif view_direction == 'PhiR':
        print('extent has not been implemented for PhiR slices')


def get_axis_labels(self, view_direction='RZ'):
    """Returns two axis label values for the x and y value. Takes
    view_direction into account."""
    if view_direction == 'RZ':
        xlabel = "R [cm]"
        ylabel = "Z [cm]"
    elif view_direction == 'PhiR':
        xlabel = "R [cm]"
        ylabel = "Phi"

    return xlabel, ylabel


def get_tallies_with_cylindrical_mesh_filters(statepoint: openmc.StatePoint):
    """scans the statepoint object to find all tallies and with cylindrical mesh
    filters, returns a list of tally indexes"""

    matching_tally_ids = []
    for tally_id, tally in statepoint.tallies.items():
        print("tally id", tally_id)
        try:
            mf = tally.find_filter(filter_type=openmc.MeshFilter)
            if isinstance(mf.mesh, openmc.CylindricalMesh):
                matching_tally_ids.append(tally.id)
                print("found regmeshfilter")
        except ValueError:
            mf = None

    return sorted(matching_tally_ids)


def get_cylindricalmesh_tallies_and_scores(statepoint: openmc.StatePoint):
    """scans the statepoint object to find all tallies and scores,
    returns list of dictionaries. Each dictionary contains tally id,
    score and tally name"""

    tallies_of_interest = get_tallies_with_cylindrical_mesh_filters(statepoint)

    tally_score_info = []
    for tally_id in tallies_of_interest:
        tally = statepoint.tallies[tally_id]
        for score in tally.scores:
            entry = {"id": tally.id, "score": score, "name": tally.name}
            tally_score_info.append(entry)

    return tally_score_info


def get_number_of_slices(self, view_direction: str):
    if view_direction == 'RZ':
        return self.dimension[1]

    elif view_direction == 'PhiR':
        return self.dimension[2]

openmc.CylindricalMesh.get_number_of_slices = get_number_of_slices
openmc.mesh.CylindricalMesh.get_number_of_slices = get_number_of_slices

openmc.CylindricalMesh.slice_of_data = slice_of_data
openmc.mesh.CylindricalMesh.slice_of_data = slice_of_data

openmc.CylindricalMesh.get_axis_labels = get_axis_labels
openmc.mesh.CylindricalMesh.get_axis_labels = get_axis_labels

openmc.CylindricalMesh.get_mpl_plot_extent = get_mpl_plot_extent
openmc.mesh.CylindricalMesh.get_mpl_plot_extent = get_mpl_plot_extent
