from typing import Dict

from geometries import TwoPileWithoutStrutsSystem
from .load import LoadCombinationResults

from .geometries import (
    SimpleLoadParams,
    ComplexLoadParams,
    TwoPileWithTwoStrutsParams,
    TwoPileWithTwoStrutsSystem,
    calculate_load_combinations,
)
from .geometries.base import BaseSystem, BeamProfileParams, CrossSectionParams, MaterialParams


class ApiError(Exception):
    pass


def beam_profile(profile_dict) -> BeamProfileParams:
    try:
        material = MaterialParams(**profile_dict["material"])
        cross_section = CrossSectionParams(**profile_dict["cross_section"])
        return BeamProfileParams(material, cross_section)
    except:
        raise ApiError("Error while decoding beam profile parameters")


def load_params(load_dict) -> SimpleLoadParams:
    try:
        return SimpleLoadParams(**load_dict)
    except:
        pass
    try:
        complex_params = ComplexLoadParams(**load_dict)
        # TODO: support complex parameters in model
    except:
        # neither simple nor complex load parameters supproted -> raise Error
        raise ApiError("Error while decoding load parameters!")
    # if we end up here, complex load parameters have been given, which are not yet supported
    raise NotImplementedError


def process_request(jsn):
    try:
        geo = jsn["geometry"]
        beam_parameters = {
            key: beam_profile(value) for key, value in jsn["beam_parameters"].items()
        }
        params = {**geo, **beam_parameters}
        load = load_params(jsn["load"])
        calc_params = jsn.get("calculation", {})
        calc_gzt = calc_params.get("gzt", True)
        calc_gzg = calc_params.get("gzg", True)
        nelem = calc_params.get("nelem", 40)

        if jsn["system"] == TwoPileWithTwoStrutsSystem.SYSTEM_IDENTIFIER:
            p = TwoPileWithTwoStrutsParams(**params)
            system = TwoPileWithTwoStrutsSystem(p, nelem=nelem)
            results = calculate_load_combinations(
                system, load, calc_gzg=calc_gzg, calc_gzt=calc_gzt
            )
            return response(system, results)
        else:
            pass
    except:
        raise ApiError()


def map_internal_forces(arr):
    return {
        "N": [v / 1000 for v in arr[0]],
        "Vz": [v / 1000 for v in arr[2]],
        "My": [v / 1000 for v in arr[4]],
    }


def map_support_forces(arr):
    return {
        "Fx": [-v / 1000 for v in reversed(arr[0])],
        "Fz": [-v / 1000 for v in reversed(arr[2])],
        "My": [-v / 1000 for v in reversed(arr[4])],
    }


def map_local_displacements(arr):
    return {
        "ux": [v for v in arr[0]],
        "uz": [v for v in arr[2]],
    }


def map_global_displacements(arr):
    return {
        "Ux": [v for v in arr[0]],
        "Uz": [v for v in arr[2]],
    }


def response(system: BaseSystem, results: Dict[str, LoadCombinationResults]):
    ret = {"results": {}}
    if "gzg" in results:
        gzg = results["gzg"]
        displacements = {
            beam: {
                "BEAM_MAXIMA": {
                    "global": map_global_displacements(
                        gzg.get_maximal_global_beam_displacements(nodes[0])
                    ),
                    "local": map_local_displacements(
                        gzg.get_maximal_local_beam_displacements(nodes[0])
                    ),
                },
                **{
                    node: {
                        "global": map_global_displacements(
                            gzg.get_maximal_global_node_displacements(node)
                        ),
                        "local": map_local_displacements(
                            gzg.get_maximal_local_node_displacements(node)
                        ),
                    }
                    for node in nodes
                },
            }
            for beam, nodes in system.get_node_uids().items()
        }
        ret["results"]["gzg"] = {"displacements": {"beams": displacements}}
    if "gzt" in results:
        gzt = results["gzt"]
        internal_forces = {
            beam: {
                "BEAM_MAXIMA": map_internal_forces(gzt.get_maximal_beam_forces(nodes[0])),
                **{node: map_internal_forces(gzt.get_maximal_node_forces(node)) for node in nodes},
            }
            for beam, nodes in system.get_node_uids().items()
        }
        support_forces = {
            node: map_support_forces(gzt.get_maximal_node_forces(node))
            for node in system.get_support_uids()
        }
        ret["results"]["gzt"] = {
            "internal_forces": {"beams": internal_forces},
            "support_forces": support_forces,
        }

    return ret
