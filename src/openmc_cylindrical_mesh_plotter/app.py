import streamlit as st
from matplotlib.colors import LogNorm
from openmc_cylindrical_mesh_plotter import (
    plot_mesh_tally_rz_slice,
    plot_mesh_tally_phir_slice,
)
import openmc


def save_uploadedfile(uploadedfile):
    with open(uploadedfile.name, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success(f"Saved File to {uploadedfile.name}")


def get_tallies_with_cylindrical_mesh_filters(statepoint: openmc.StatePoint):
    """scans the statepoint object to find all tallies and with cylindrical mesh
    filters, returns a list of tally indexes"""

    matching_tally_ids = []
    for _, tally in statepoint.tallies.items():
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


def header():
    """This section writes out the page header common to all tabs"""

    st.set_page_config(
        page_title="OpenMC Cylindrical Mesh Plotter",
        page_icon="⚛",
        layout="wide",
    )

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {
                    visibility: hidden;
                    }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.write(
        """
            # OpenMC Cylindrical Mesh Plotter

            ### ⚛ A plotting user interface for cylindrical meshes.

            🐍 Run this app locally with Python ```pip install openmc_cylindrical_mesh_plotter``` then run with ```openmc_cylindrical_mesh_plotter```

            ⚙ Produce MatPlotLib plots in batch with the 🐍 [Python API](https://github.com/fusion-energy/openmc_cylindrical_mesh_plotter/tree/master/examples)

            💾 Raise a feature request, report and issue or make a contribution on [GitHub](https://github.com/fusion-energy/openmc_cylindrical_mesh_plotter)

            📧 Email feedback to mail@jshimwell.com

            🔗 This package forms part of a more [comprehensive openmc plot](https://github.com/fusion-energy/openmc_plot) package where geometry, tallies, slices, etc can be plotted and is hosted on [xsplot.com](https://www.xsplot.com/) .
        """
    )

    st.write("<br>", unsafe_allow_html=True)


def main():
    st.write(
        """
            👉 Carry out an OpenMC simulation to generate a ```statepoint.h5``` file.

        """
        # Not got a h5 file handy, right mouse 🖱️ click and save these links
        # [ example 1 ](https://fusion-energy.github.io/openmc_cylindrical_mesh_plotter/examples/csg_tokamak/geometry.xml),
        # [ example 2 ](https://fusion-energy.github.io/openmc_cylindrical_mesh_plotter/examples/csg_cylinder_box/geometry.xml)
    )

    statepoint_file = st.file_uploader("Select your statepoint h5 file", type=["h5"])

    if statepoint_file is None:
        new_title = '<center><p style="font-family:sans-serif; color:Red; font-size: 30px;">Select your depletion results h5 file</p></center>'
        st.markdown(new_title, unsafe_allow_html=True)

    else:
        save_uploadedfile(statepoint_file)
        statepoint = openmc.StatePoint(statepoint_file.name)

        tally_description = get_cylindricalmesh_tallies_and_scores(statepoint)
        tally_description_str = [
            f"ID={td['id']} score={td['score']} name={td['name']}"
            for td in tally_description
        ]

        tally_description_to_plot = st.sidebar.selectbox(
            label="Tally to plot", options=tally_description_str, index=0
        )
        tally_id_to_plot = tally_description_to_plot.split(" ")[0][3:]
        score = tally_description_to_plot.split(" ")[1][6:]

        basis = st.sidebar.selectbox(
            label="view direction",
            options=("PhiR", "RZ"),
            index=0,
            key="axis",
            help="",
        )

        value = st.sidebar.radio("Tally mean or std dev", options=["mean", "std_dev"])

        axis_units = st.sidebar.selectbox(
            "Axis units", ["km", "m", "cm", "mm"], index=2
        )

        volume_normalization = st.sidebar.radio(
            "Divide value by mesh voxel volume", options=[True, False]
        )

        scaling_factor = st.sidebar.number_input(
            "Scaling factor",
            value=1.0,
            help="Input a number that will be used to scale the mesh values. For example a input of 2 would double all the values.",
        )

        tally = statepoint.get_tally(id=int(tally_id_to_plot))

        mesh = tally.find_filter(filter_type=openmc.MeshFilter).mesh

        if basis == "RZ":
            max_value = int(tally.shape[1] / 2)  # index 1 is the phi value
        if basis == "PhiR":
            max_value = int(tally.shape[2] / 2)  # index 2 is the z value

        if max_value == 0:
            slice_index = 0
        else:
            slice_index = st.sidebar.slider(
                label="slice index",
                min_value=0,
                value=int(max_value / 2),
                max_value=max_value,
            )

        # contour_levels_str = st.sidebar.text_input(
        #     "Contour levels",
        #     help="Optionally add some comma deliminated contour values",
        # )

        # if contour_levels_str:
        #     contour_levels = sorted(
        #         [float(v) for v in contour_levels_str.strip().split(",")]
        #     )
        # else:
        #     contour_levels = None

        colorbar = st.sidebar.radio("Include colorbar", options=[True, False])

        title = st.sidebar.text_input(
            "Colorbar title",
            help="Optionally set your own colorbar label for the plot",
            value="colorbar title",
        )

        log_lin_scale = st.sidebar.radio("Scale", options=["log", "linear"])
        if log_lin_scale == "linear":
            norm = None
        else:
            norm = LogNorm()

        if basis == "RZ":
            plot = plot_mesh_tally_rz_slice(
                tally=tally,
                slice_index=slice_index,
                score=score,
                axes=None,
                axis_units=axis_units,
                value=value,
                # outline: bool = False,
                # outline_by: str = "cell",
                # geometry: Optional["openmc.Geometry"] = None,
                # geometry_basis: str = "xz",
                # pixels: int = 40000,
                colorbar=colorbar,
                volume_normalization=volume_normalization,
                scaling_factor=scaling_factor,
                colorbar_kwargs={"label": title},
                norm=norm
                # outline_kwargs: dict = _default_outline_kwargs,
                # **kwargs,
            )
        elif basis == "PhiR":
            plot = plot_mesh_tally_phir_slice(
                tally=tally,  # "openmc.Tally",
                slice_index=slice_index,  # Optional[int] = None,
                score=score,  # Optional[str] = None,
                # axes,# Optional[str] = None,
                axis_units=axis_units,  # str = "cm",
                value=value,  # str = "mean",
                # outline,# bool = False,
                # outline_by,# str = "cell",
                # geometry,# Optional["openmc.Geometry"] = None,
                # pixels,# int = 40000,
                colorbar=colorbar,  # bool = True,
                volume_normalization=volume_normalization,  # bool = True,
                scaling_factor=scaling_factor,  # Optional[float] = None,
                colorbar_kwargs={"label": title},
                norm=norm,
                # outline_kwargs,# d
            )

        plot.figure.savefig("openmc_plot_cylindricalmesh_image.png")
        st.pyplot(plot.figure)
        with open("openmc_plot_cylindricalmesh_image.png", "rb") as file:
            st.download_button(
                label="Download image",
                data=file,
                file_name="openmc_plot_cylindricalmesh_image.png",
                mime="image/png",
            )


if __name__ == "__main__":
    header()
    main()
