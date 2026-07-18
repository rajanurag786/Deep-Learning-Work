SUPPORTED_SHAPES = {"box", "cylinder"}
SUPPORTED_MATERIALS = {"steel", "concrete"}
SUPPORTED_ANALYSES = {"static", "static_structural"}

def validate_simulation_input(parsed_data: dict) -> dict:
    if not isinstance(parsed_data, dict):
        raise ValueError("The parsed input must be dictionary")

    geometry = parsed_data.get("geometry")
    if geometry is None:
        raise ValueError("The input geometry is required")

    if isinstance(geometry, list):
        if len(geometry) == 0:
            raise ValueError("Need at least 1 geometry.")
        geometries = geometry
    elif isinstance(geometry, dict):
        geometries = [geometry]
    else:
        raise ValueError("Geometry must be a dictionary or list of dictionaries.")

    for geometry in geometries:
        shape = geometry.get("type")
        if shape not in SUPPORTED_SHAPES:
            raise ValueError(f"Unsupported geometry type: {shape}")

        params = geometry.get("parameters", {})

        required = ["width", "depth", "height"] if shape == "box" else ["radius", "height"]

        for key in required:
            if key not in params:
                raise ValueError(f"Missing geometry parameter: {key}")
            if float(params[key]) <= 0:
                raise ValueError(f"{key} must be greater than zero.")

    load = parsed_data.get("load")
    if not load or "magnitude_N" not in load:
        raise ValueError("Missing load.magnitude_N.")

    if float(load["magnitude_N"]) <= 0:
        raise ValueError("Load magnitude must be greater than zero.")

    material = parsed_data.get("material", params.get("material", "steel")).lower()
    if material not in SUPPORTED_MATERIALS:
        raise ValueError(f"Unsupported material: {material}")

    return parsed_data
