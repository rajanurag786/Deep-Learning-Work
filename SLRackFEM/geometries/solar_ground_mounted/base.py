import math
from dataclasses import dataclass
from typing import Dict, List

from ..base import BaseSystem, BeamProfileParams
from .constants import *


class PurlinSystem(BaseSystem):
    def get_node_uids(self) -> Dict[str, List[str]]:
        uids = super().get_node_uids()
        uids[B_GIRDER] = [P_GIRDER_LEFT, P_GIRDER_RIGHT]
        return uids


@dataclass
class GroundMountedSolarParams:
    """Basic parameters for a ground mounted system

    Args:
        rigid_supports (bool): Whether the supports are assumed to be rigid for rotation around Y axis
        alpha (float): Ground slope angle (degrees)
        beta (float): Angle of inclination of the solar panels (degrees):
        t (float): anchoring depth of the pile below ground
        h (float): height in meters of the left (or only) pile above ground
        o_l (float): overhang in meters on the left side of the left (or only) pile (projected on x axis)
        o_r (float): overhang in meters on the right side of the right (or only) pile (projected on x axis)
        pile (BeamProfileParams): cross section parameters for the pile
        girder (BeamProfileParams): cross section parameters for the girder
    """

    rigid_supports: bool
    alpha: float
    beta: float
    t: float
    h: float
    o_l: float
    o_r: float
    pile: BeamProfileParams
    girder: BeamProfileParams

    @property
    def alpha_rad(self):
        return math.radians(self.alpha)

    @property
    def beta_rad(self):
        return math.radians(self.beta)
