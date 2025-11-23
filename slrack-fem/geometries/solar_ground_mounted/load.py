from typing import Dict
import math
from dataclasses import dataclass

from ...load import *
from .base import PurlinSystem
from .constants import *


@dataclass
class LoadParams:
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
    Wdown: float
    Wup: float


def calculate_system(
    system: PurlinSystem, load: LoadParams, calc_gzg: bool = True, calc_gzt: bool = True
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
    # TODO: extract worker_processes parameter to config
    gzt, gzg = calc_load_combinations(system, gzt, gzg, worker_processes=4)
    return {
        "gzt": gzt,
        "gzg": gzg,
    }
