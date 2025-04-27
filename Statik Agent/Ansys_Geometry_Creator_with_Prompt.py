from ansys.mapdl.core import launch_mapdl

def assign_material(mapdl, material_name):
    if material_name.lower() == "concrete":
        mapdl.mp("EX", 1, 30e9)
        mapdl.mp("NUXY", 1, 0.2)
    elif material_name.lower() == "steel":
        mapdl.mp("EX", 1, 200e9)
        mapdl.mp("NUXY", 1, 0.3)
    else:
        raise ValueError(f"Unknown material: {material_name}")

def run_static_analysis(parsed_data):
    geometry = parsed_data["geometry"]
    height = geometry["height_m"]
    size = geometry["size_m"]
    load = parsed_data["load"]["magnitude_N"]

    # Launch MAPDL explicitly
    mapdl = launch_mapdl(exec_file=r"C:\Program Files\ANSYS Inc\ANSYS Student\v251\ansys\bin\winx64\ANSYS251.exe")
    mapdl.clear()
    mapdl.prep7()

    geometry_list = parsed_data["geometry"]
    z_offset = 0
    for i, geo in enumerate(geometry_list):
        shape = geo["type"]
        params = geo["parameters"]
        material = params.get("material", "steel")
        height = params["height"]

        if shape == "cylinder":
            radius = params["radius"]
            mapdl.cylind(0, radius, 0, height)
        elif shape == "box":
            w, d = params["width"], params["depth"]
            mapdl.block(0, w, 0, d, 0, height)
        else:
            raise ValueError(f"Unsupported geometry type: {shape}")

        # Move if not the base
        if i > 0 and geo.get("relationship", "") == "on top of":
            mapdl.vtran("ALL", 0, 0, z_offset)

        z_offset += height  # stack next object higher

    # # 2. Define material properties (example: steel)
    # mapdl.mp('EX', 1, 2e11)  # Young's Modulus in Pa
    # mapdl.mp('NUXY', 1, 0.3)  # Poisson's Ratio

    material = geometry_list[0]["parameters"].get("material", "steel")
    assign_material(mapdl, material)

    # 3. Define element type
    mapdl.et(1, 'SOLID185')

    # 4. Mesh the volume
    mapdl.vsweep('ALL')
    mapdl.esize(size / 5)  # element size
    mapdl.vmesh('ALL')
    mapdl.eplot()

    # 5. Apply constraints (fix bottom face, Z=0)
    mapdl.nsel('S', 'LOC', 'Z', 0)
    mapdl.d('ALL', 'ALL', 0)
    mapdl.allsel()

    # 6. Apply axial force on top face (Z = height)
    mapdl.nsel('S', 'LOC', 'Z', height)
    mapdl.f('ALL', 'FZ', -load)  # axial load in negative Z
    mapdl.allsel()

    # 7. Solve
    mapdl.run('/SOLU')
    mapdl.antype('STATIC')
    mapdl.solve()
    mapdl.finish()

    # 8. Post-processing
    mapdl.post1()
    mapdl.set(1)
    # help(mapdl.post_processing)

    # Max displacement in Z
    max_disp_z = mapdl.post_processing.nodal_displacement("Z").max()
    plot_disp = mapdl.post_processing.plot_nodal_displacement("Z")

    # Max equivalent (von Mises) stress
    # max_stress_eqv = mapdl.post_processing.plot_nodal_eqv_stress("EQV")

    print(f"Max Displacement (Z): {max_disp_z:.6f} m")
    # if max_stress_eqv is not None:
    #     print(f"Max Von Mises Stress: {max_stress_eqv:.6f} Pa")

    return {
        "max_displacement_z": float(max_disp_z)
    }


# Example parsed prompt data
parsed_prompt = {
    "geometry": {
        "type": "column",
        "height_m": 0.25,
        "size_m": 0.015
    },
    "load": {
        "type": "axial",
        "magnitude_N": 100000000
    },
    "analysis_type": "static",
    "outputs": {
        "graphs": [],
        "images": []
    }
}

# results = run_static_analysis(parsed_prompt)
