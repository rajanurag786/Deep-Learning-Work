# Ansys_Geometry_Generator.py
from ansys.mapdl.core import launch_mapdl
import pandas as pd
import os

ANSYS_EXEC_PATH = os.getenv(
    "ANSYS_EXEC_PATH",
    r"D:\ANSYS Inc\ANSYS Student\v261\ansys\bin\winx64\ANSYS261.exe"
)

MATERIAL_DB = {
    "steel": {
        "E": 200e9,
        "nu": 0.30,
        "density": 7850,
        "yield_strength": 250e6,
    },
    "concrete": {
        "E": 30e9,
        "nu": 0.20,
        "density": 2400,
        "compressive_strength": 30e6,
    },
}


def assign_material(mapdl, material_name: str):
    material_key = material_name.lower()

    if material_key not in MATERIAL_DB:
        raise ValueError(f"Unsupported material: {material_name}")

    mat = MATERIAL_DB[material_key]

    mapdl.mp("EX", 1, mat["E"])
    mapdl.mp("NUXY", 1, mat["nu"])
    mapdl.mp("DENS", 1, mat["density"])

    return mat

def create_box(mapdl, params, x0, y0, z0):
    width = float(params["width"])
    depth = float(params["depth"])
    height = float(params["height"])

    if width <= 0 or depth <= 0 or height <= 0:
        raise ValueError("Box dimensions must be greater than zero.")

    mapdl.block(
        x0,
        x0 + width,
        y0,
        y0 + depth,
        z0,
        z0 + height,
    )

    return {
        "type": "box",
        "xmin": x0,
        "xmax": x0 + width,
        "ymin": y0,
        "ymax": y0 + depth,
        "zmin": z0,
        "zmax": z0 + height,
        "height": height,
    }

def create_cylinder(mapdl, params, x0, y0, z0):
    radius = float(params["radius"])
    height = float(params["height"])

    if radius <= 0 or height <= 0:
        raise ValueError("Cylinder radius and height must be greater than zero.")

    mapdl.cyl4(
        x0,
        y0,
        radius,
        depth=height,
    )

    new_vol = int(mapdl.get("NEWVOL", "VOLU", 0, "NUM", "MAX"))

    if z0 != 0:
        mapdl.vsel("S", "VOLU", "", new_vol)
        mapdl.vmove("ALL", "", "", "", "", "", "", z0)
        mapdl.allsel()


    return {
        "type": "cylinder",
        "xmin": x0 - radius,
        "xmax": x0 + radius,
        "ymin": y0 - radius,
        "ymax": y0 + radius,
        "zmin": z0,
        "zmax": z0 + height,
        "height": height,
    }

def create_geometry(mapdl, geometry_item):
    shape_type = geometry_item["type"].lower()
    params = geometry_item.get("parameters", {})

    position = params.get("position", [0.0, 0.0, 0.0])
    x0, y0, z0 = position

    if shape_type == "box":
        return create_box(mapdl, params, x0, y0, z0)

    elif shape_type == "cylinder":
        return create_cylinder(mapdl, params, x0, y0, z0)

    else:
        raise ValueError(f"Unsupported geometry type: {shape_type}")

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
    if isinstance(geom, dict):
        geometry_list = [geom]
    elif isinstance(geom, list):
        geometry_list = geom
    else:
        raise ValueError(f"Geometry must be a dict or list, got {type(geom)}")

    if not geometry_list:
        raise ValueError("No geometry found in the parsed data.")

    # Uses the first geometry for fallback material lookup.
    shape_def = geometry_list[0]
    params = shape_def.get("parameters", {})


    # 4) Extract load
    # It may live at the top level, inside the geometry dict, or inside the parameters dict.
    load_dict = parsed_data.get("load") or \
                shape_def.get("load") or \
                params.get("load", {})
    if not load_dict or "magnitude_N" not in load_dict:
        raise ValueError(
            "No 'load.magnitude_N' found in parsed data. "
            f"Received parsed_data: {parsed_data}"
        )
    load = float(load_dict["magnitude_N"])

    # 5) Material (may live in params or at top level)
    material = (
            parsed_data.get("material")
            or params.get("material")
            or "steel"
    )

    # 6) Height (for boundary & load location)
    # h = float(params.get("height") or params.get("length") or 1.0)

    # -- Start MAPDL --
    mapdl = launch_mapdl(exec_file=ANSYS_EXEC_PATH)

    try:

        mapdl.clear()
        mapdl.prep7()

        # -- Create geometry --
        # if shape == "cylinder":
        #     r = float(params["radius"])
        #     mapdl.cylind(0, r, 0, h)
        # elif shape == "box":
        #     w = float(params["width"])
        #     d = float(params["depth"])
        #     mapdl.block(0, w, 0, d, 0, h)
        # else:
        #     raise ValueError(f"Unsupported shape: {shape}")
        created_geometries = []

        for geometry_item in geometry_list:
            created = create_geometry(mapdl, geometry_item)
            created_geometries.append(created)

        mapdl.allsel()

        # Glue touching volumes so stacked/connected objects behave as one body.
        if len(created_geometries) > 1:
            try:
                mapdl.vglue("ALL")
            except Exception:
                mapdl.allsel()
                mapdl.vovlap("ALL")
            mapdl.allsel()

        global_xmin = min(g["xmin"] for g in created_geometries)
        global_xmax = max(g["xmax"] for g in created_geometries)
        global_ymin = min(g["ymin"] for g in created_geometries)
        global_ymax = max(g["ymax"] for g in created_geometries)
        global_zmin = min(g["zmin"] for g in created_geometries)
        global_zmax = max(g["zmax"] for g in created_geometries)

        model_size = max(
            global_xmax - global_xmin,
            global_ymax - global_ymin,
            global_zmax - global_zmin,
        )

        tol = max(model_size * 1e-5, 1e-6)


        # 4) Material, element, and mesh
        material_props = assign_material(mapdl, material)
        mapdl.et(1, "SOLID187")  # tetrahedral solid element
        mapdl.mshape(1, "3D")  # tetrahedral volume mesh
        mapdl.mshkey(0)  # free meshing

        # Mesh size based on total model size.
        mesh_size = model_size / 10

        if mesh_size <= 0:
            raise RuntimeError("Invalid mesh size computed from model dimensions.")

        mapdl.esize(mesh_size)
        mapdl.vmesh("ALL")

        num_nodes = len(mapdl.mesh.nnum)
        num_elements = len(mapdl.mesh.enum)

        if num_nodes == 0 or num_elements == 0:
            raise RuntimeError("Meshing failed. No nodes or elements were generated.")

        # Boundary condition: fix global bottom face
        mapdl.nsel("S", "LOC", "Z", global_zmin - tol, global_zmin + tol)

        bottom_nodes = mapdl.mesh.nnum
        if len(bottom_nodes) == 0:
            raise RuntimeError("No bottom-face nodes found for support application.")

        mapdl.d("ALL", "ALL", 0)
        mapdl.allsel()

        # 6) Load on top face at Z = height
        mapdl.nsel("S", "LOC", "Z", global_zmax - tol, global_zmax + tol)

        top_nodes = mapdl.mesh.nnum
        if len(top_nodes) == 0:
            raise RuntimeError("No top-face nodes found for load application.")

        force_per_node = -load / len(top_nodes)
        mapdl.f("ALL", "FZ", force_per_node)

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



        # 9) Return results + plot paths
        return {
            "material_properties": material_props,
            "num_nodes": int(num_nodes),
            "num_elements": int(num_elements),

            "geometry_count": len(created_geometries),
            "geometry_summary": created_geometries,
            "global_bounds": {
                "xmin": global_xmin,
                "xmax": global_xmax,
                "ymin": global_ymin,
                "ymax": global_ymax,
                "zmin": global_zmin,
                "zmax": global_zmax,
            },

            "max_displacement_z": float(abs(disp_z).max()),
            "min_displacement_z": float(disp_z.min()),
            "max_displacement_z_signed": float(disp_z.max()),
            "max_stress": float(stress.max()),

            "stress_img": stress_img,
            "disp_img": disp_img,
            "def_img": def_img,
            "disp_side": disp_side,
            "stress_side": stress_side
        }
    finally:
        mapdl.exit()


