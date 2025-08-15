# Ansys_Geometry_Generator.py
from ansys.mapdl.core import launch_mapdl
import matplotlib.pyplot as plt
import pandas as pd

def assign_material(mapdl, material_name: str):
    m = material_name.lower()
    if m == "concrete":
        mapdl.mp("EX", 1, 30e9)
        mapdl.mp("NUXY", 1, 0.2)
    else:  # default steel
        mapdl.mp("EX", 1, 2e11)
        mapdl.mp("NUXY", 1, 0.3)

def run_static_analysis(parsed_data):
    """
    parsed_data may be either:
      - {'geometry': {...}, 'load': {...}, ...}
      - {'type': 'cylinder', 'parameters': {...}, 'load': {...}, ...}
      - [ {...} ]  # list with one of the above dicts
    This function normalizes and runs the MAPDL workflow.
    """
    # Pull out geometry
    geom = parsed_data.get("geometry")
    if geom is None:
        raise ValueError("No geometry provided in parsed_data.")

    # Normalize to a list of shape definitions
    if isinstance(geom, dict):
        geometry_list = [geom]
    elif isinstance(geom, list):
        geometry_list = geom
    else:
        raise ValueError(f"Geometry must be a dict or list, got {type(geom)}")

    # Enforce exactly one primitive (for now)
    if len(geometry_list) != 1:
        raise ValueError(f"Expected 1 geometry, got {len(geometry_list)}")

    # Now safe to unpack
    shape_def = geometry_list[0]
    shape = shape_def["type"]
    params = shape_def["parameters"]
    # 1) Normalize list → dict
    # if isinstance(parsed_data, list):
    #     if len(parsed_data) != 1:
    #         raise ValueError(f"Expected 1 entry, got {len(parsed_data)}")
    #     parsed_data = parsed_data[0]
    #
    # # 2) Extract the geometry block
    # if "geometry" in parsed_data:
    #     geom = parsed_data["geometry"]
    #     # geometry itself might be a list or dict
    #     if isinstance(geom, list):
    #         if len(geom) != 1:
    #             raise ValueError(f"Expected 1 geometry, got {len(geom)}")
    #         geom = geom[0]
    # else:
    #     # assume parsed_data *is* the geometry block
    #     geom = parsed_data
    #
    # # 3) Now geom must be a dict with keys: 'type', 'parameters'
    # if not isinstance(geom, dict) or "type" not in geom or "parameters" not in geom:
    #     raise ValueError(f"Cannot find geometry type/parameters in:\n{geom}")
    #
    # shape  = geom["type"]
    # params = geom["parameters"]

    # 4) Extract load
    # It may live at parsed_data["load"] or inside geom["load"]
    load_dict = parsed_data.get("load") or geom.get("load", {})
    if not load_dict or "magnitude_N" not in load_dict:
        raise ValueError("No load.magnitude_N found in parsed data.")
    load = float(load_dict["magnitude_N"])

    # 5) Material (may live in params or at top level)
    material = params.get("material") or parsed_data.get("material", "steel")

    # 6) Height (for boundary & load location)
    h = float(params.get("height") or params.get("length") or 1.0)

    # -- Start MAPDL --
    mapdl = launch_mapdl(exec_file=r"C:\Program Files\ANSYS Inc\ANSYS Student\v251\ansys\bin\winx64\ANSYS251.exe")
    mapdl.clear(); mapdl.prep7()

    # -- Create geometry --
    if shape == "cylinder":
        r = float(params["radius"])
        mapdl.cylind(0, r, 0, h)
    elif shape == "box":
        w = float(params["width"])
        d = float(params["depth"])
        mapdl.block(0, w, 0, d, 0, h)
    else:
        raise ValueError(f"Unsupported shape: {shape}")


    # 4) Material, element, and mesh
    assign_material(mapdl, material)
    mapdl.et(1, "SOLID185")
    mapdl.vsweep("ALL")
    # choose an element size based on geometry
    if shape == "cylinder":
        base = r
    else:
        base = min(w, d)
    mapdl.esize(base / 5)
    mapdl.vmesh("ALL")

    # 5) Boundary conditions: fix bottom (Z=0)
    mapdl.nsel("S", "LOC", "Z", 0)
    mapdl.d("ALL", "ALL", 0)
    mapdl.allsel()

    # 6) Load on top face at Z = height
    mapdl.nsel("S", "LOC", "Z", h)
    mapdl.f("ALL", "FZ", -load)
    mapdl.allsel()

    # 7) Solve
    mapdl.run("/SOLU")
    mapdl.antype("STATIC")
    mapdl.solve()
    mapdl.finish()

    # 8) Post-processing
    mapdl.post1()
    mapdl.set(1)

    # ensure global Cartesian
    mapdl.run("CSYS,0")

    # rotate 90° about the X-axis so Z is up in the screen
    # this PROFILE view shows the full length
    mapdl.run("/VIEW,1,0,90,0")

    # now plot the von Mises stress on that side view
    mapdl.run("PLNSOL, S, EQV")
    stress_side = "stress_side.png"
    mapdl.screenshot(stress_side)

    # rotate back or leave—then do displacement
    mapdl.run("PLNSOL, UZ")  # or PDISP,2 for Z-disp contour
    disp_side = "disp_side.png"
    mapdl.screenshot(disp_side)

    # 1) Stress contour (von Mises)
    mapdl.run("PLNSOL, S, EQV")  # element solution contour for EQV stress
    stress_img = "stress_contour.png"
    mapdl.screenshot(stress_img)  # saves current view

    # 2) Displacement contour (Z component)
    mapdl.run("PLDISP, 1")  # plot displacement, component 2 = Z translational
    disp_img = "disp_contour.png"
    mapdl.screenshot(disp_img)

    # 3) Deformed shape overlay (optional)
    mapdl.run("PLDISP, 1")                         # total magnitude (component 3)
    def_img = "deformed_shape.png"
    mapdl.screenshot(def_img)


    disp_z = mapdl.post_processing.nodal_displacement("Z")
    stress = mapdl.post_processing.nodal_stress_intensity()

    # extract coords and node IDs from numpy arrays
    coords = mapdl.mesh.nodes  # shape (n_nodes, 3)
    node_ids = mapdl.mesh.nnum

    # build a DataFrame manually
    nodes_df = pd.DataFrame({
        "node": node_ids,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "disp_z": disp_z,
        "stress": stress
    })

    # sort by height for plotting
    sorted_df = nodes_df.sort_values("z")

    # # 7) Plot Displacement vs Height
    # plt.figure()
    # plt.plot(sorted_df["z"], sorted_df["disp_z"])
    # plt.xlabel("Height (m)")
    # plt.ylabel("Displacement Z (m)")
    # plt.title("Displacement vs Height")
    # disp_plot = "disp_vs_z.png"
    # plt.savefig(disp_plot)
    # plt.close()
    #
    # # 8) Plot Stress vs Height
    # plt.figure()
    # plt.plot(sorted_df["z"], sorted_df["stress"])
    # plt.xlabel("Height (m)")
    # plt.ylabel("Von Mises Stress (Pa)")
    # plt.title("Stress vs Height")
    # stress_plot = "stress_vs_z.png"
    # plt.savefig(stress_plot)
    # plt.close()

    # 9) Return results + plot paths
    return {
        "max_displacement_z": float(disp_z.max()),
        "max_stress": float(stress.max()),
        "stress_img": stress_img,
        "disp_img": disp_img,
        "def_img": def_img,
        "disp_side": disp_side,
        "stress_side": stress_side
    }

    # return {
    #     "max_displacement_z": float(max_disp),
    #     "max_stress": float(max_stress)
    # }
