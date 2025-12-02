from typing import Dict, List
import math
from dataclasses import dataclass

from ...load import *
from .base import PurlinSystem
from .constants import *


@dataclass
class BaseLoadParams:
    """Basic load parameters for solar ground mounted systems

    Args:
        e (float): Distance between supports
        Gk (float): Self weight of panels (kN/m^2)
        Sk (float): Snow load on panels (kN/m^2)
        Wdown (float): Wind pressure load on panels (kN/m^2)
        Wup (float): Wind suction load on panels (kN/m^2)
    """

    e: float
    Gk: float
    Sk: float


@dataclass
class SimpleLoadParams(BaseLoadParams):
    """Load parameters for solar ground mounted systems with simple wind load

    Args:
        Wdown (float): Wind pressure load on panels (kN/m^2)
        Wup (float): Wind suction load on panels (kN/m^2)
    """

    Wdown: float
    Wup: float

    # validation
    def __post_init__(self):
        if not all(isinstance(obj, float) for obj in [self.Wdown, self.Wup]):
            raise ValueError


@dataclass
class ComplexLoadParams(BaseLoadParams):
    """Load parameters for solar ground mounted systems with complex wind load

    Args:
        Wdown (List[Dict[str, float]]): Wind pressure load on panels (kN/m^2)
        Wup (List[Dict[str, float]]): Wind suction load on panels (kN/m^2)
    """

    Wdown: List[Dict[str, float]]
    Wup: List[Dict[str, float]]

    # validation
    def __post_init__(self):
        if not all(
            isinstance(obj, list)
            and all(
                isinstance(inner, dict) and "fraction" in inner and "load" in inner for inner in obj
            )
            and math.isclose(sum(inner["fraction"] for inner in obj), 1.0)
            for obj in [self.Wdown, self.Wup]
        ):
            raise ValueError


def calculate_load_combinations(
    system: PurlinSystem, load: SimpleLoadParams, calc_gzg: bool = True, calc_gzt: bool = True) -> Dict[str, LoadCombinationResults]:
    """Calculate load combinations
    Args:
    """
    Wdown = LoadCollection(
        {
            B_GIRDER: []
        }
    )
    Wup = LoadCollection(
        {
            B_GIRDER: []
        }
    )
def calculate_system(
    system: PurlinSystem, load: BaseLoadParams, calc_gzg: bool = True, calc_gzt: bool = True
) -> Dict[str, LoadCombinationResults]:
    e = load.e

    ## loads
    self_weight = LoadCollection(
        {
            B_GIRDER: [
                e * DL.new(Fz=load.Gk * 1000),
            ],
        },
        dead_weight=True,
    )

    cos_beta = math.cos(system.params.beta_rad)
    snow = LoadCollection(
        {
            B_GIRDER: [
                e * DL.new(Fz=cos_beta * load.Sk * 1000),
            ]
        }
    )

    wind_down = LoadCollection(
        {
            B_GIRDER: [
                e * DL.new(Fz=load.Wdown * 1000),
            ]
        }
    )
    wind_up = LoadCollection(
        {
            B_GIRDER: [
                e * DL.new(Fz=load.Wup * 1000),
            ]
        }
    )

    gzt = LoadCombination(
        [
            LoadCollection(
                [
                    1.35 * self_weight,
                    1.5 * snow,
                    1.5 * 0.6 * wind_down,
                ]
            ),
            LoadCollection(
                [
                    1.35 * self_weight,
                    1.5 * 0.5 * snow,
                    1.5 * wind_down,
                ]
            ),
            LoadCollection(
                [
                    0.9 * self_weight,
                    1.5 * wind_up,
                ]
            ),
        ]
    )

    gzg = LoadCombination(
        [
            LoadCollection(
                [
                    self_weight,
                    snow,
                    0.6 * wind_down,
                ]
            ),
            LoadCollection(
                [
                    self_weight,
                    0.5 * snow,
                    wind_down,
                ]
            ),
            LoadCollection(
                [
                    self_weight,
                    wind_up,
                ]
            ),
        ]
    )

    # TODO: respect calc_gzg; calc_gzt parameters
    load_combo = []
    results = []

    if calc_gzt:
        load_combo.append(gzt)
        results.append('gzt')

    if calc_gzg:
        load_combo.append(gzg)
        results.append('gzg')

    results_dict: Dict[str, LoadCombinationResults] = {"gzt": None, "gzg": None}

    if load_combo:
        lc_results = calc_load_combinations(system, *load_combo, worker_processes=4)
        for key, res in zip(results, lc_results):
            results_dict[key] = res

    return results_dict
    # TODO: extract worker_processes parameter to config
    # gzt, gzg = calc_load_combinations(system, gzt, gzg, worker_processes=4)
    # return {
    #     "gzt": gzt,
    #     "gzg": gzg,
    # }
