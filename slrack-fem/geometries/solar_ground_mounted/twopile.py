import math
from dataclasses import dataclass
from typing import Dict, List

from src.geometries.solar_ground_mounted.base import PurlinSystem

from ..base import BeamProfileParams
from .base import GroundMountedSolarParams

from ...lib import *
from ...model import Model, Node, Beam
from .constants import *


@dataclass
class TwoPileParams(GroundMountedSolarParams):
    """Basic parameters for a two pile system

    Args:
        a (float): distance between left and right pile
    """

    a: float

    # height of right (longer) pile
    def get_h2(self):
        return self.h + self.a * math.tan(self.alpha_rad) - self.a * math.tan(self.beta_rad)

    # height of girder above ground
    def get_h7(self):
        return (
            self.h
            + (self.a + self.o_r) * math.tan(self.alpha_rad)
            - (self.a + self.o_r) * math.tan(self.beta_rad)
        )


@dataclass
class TwoPileWithTwoStrutsParams(TwoPileParams):
    """Parameters for two pile with one strut on either side

    Args:
        s_l (float): distance between left pile/girder joint and left strut/girder joint and, accordingly, strut/pile joint
        s_r (float): distance between right pile/girder joint and left strut/girder joint and, accordingly, strut/pile joint
        strut (BeamProfileParams): cross section parameters for the struts
    """

    s_l: float
    s_r: float
    strut: BeamProfileParams

    def len_left_strut(self):
        return 2 * self.s_l * math.cos(math.pi / 4 - self.beta_rad / 2)

    def len_right_strut(self):
        return 2 * self.s_r * math.cos(math.pi / 4 + self.beta_rad / 2)


# TODO: introduce TwoPileSystem (without struts)
class TwoPileWithTwoStrutsSystem(PurlinSystem):
    def __init__(self, params: TwoPileWithTwoStrutsParams, *args, **kwargs):
        super().__init__(params, *args, **kwargs)

    def create_model(self) -> Model:
        zero = Vector.xz(0, 0)
        b_rad = self.params.beta_rad
        a_rad = self.params.alpha_rad
        U = self.params.o_l
        U_r = self.params.o_r
        T = self.params.t
        H = self.params.h
        A = self.params.a
        S_L = self.params.s_l
        S_R = self.params.s_r

        vec_girder = ParametricVector.xz(lambda x: x, lambda x: x * math.tan(b_rad))

        model = super().create_model()
        beam_profile_pile = self.add_beam_profile(model, self.params.pile)
        beam_profile_girder = self.add_beam_profile(model, self.params.girder)
        beam_profile_strut = self.add_beam_profile(model, self.params.strut)

        strut1_girder = vec_girder(-math.cos(b_rad) * S_L)
        girder = Beam(
            beam_profile_girder,
            [
                Node(vec_girder(-U), "1"),
                Node(strut1_girder, "2"),
                Node(zero, "3"),
                Node(vec_girder(A / 2), "4"),
                Node(vec_girder(A), "5"),
                Node(vec_girder(A + math.cos(b_rad) * S_R), "6"),
                Node(vec_girder(A + U_r), "7"),
            ],
            name=B_GIRDER,
        )
        model.add_beam(girder)

        pile1 = Beam(
            beam_profile_pile,
            [
                Node(Vector.xz(0, H + T), "A1"),
                Node(Vector.xz(0, S_L), "10"),
                Node(zero, "p1_top"),
            ],
            up_direction=[1, 0, 0],
            name=B_PILE_LEFT,
        )
        model.add_beam(pile1)

        p2_top = vec_girder(A)
        pile2 = Beam(
            beam_profile_pile,
            [
                Node(Vector.xz(A, H + T + A * math.tan(a_rad)), "A2"),
                Node(p2_top + [0, 0, S_R], "11"),
                Node(p2_top, "p2_top"),
            ],
            name=B_PILE_RIGHT,
        )
        model.add_beam(pile2)

        end = Vector.xz(0, S_L)
        strut1 = Beam(
            beam_profile_strut,
            [
                Node(strut1_girder, "s1_top"),
                Node((strut1_girder + end) * 0.5, "8"),
                Node(Vector.xz(0, S_L), "s1_bottom"),
            ],
            name=B_STRUT_LEFT,
        )
        model.add_beam(strut1)

        strut2_girder = vec_girder(A + math.cos(b_rad) * S_R)
        end2 = p2_top + [0, 0, S_R]
        strut2 = Beam(
            beam_profile_strut,
            [
                Node(strut2_girder, "s2_top"),
                Node((strut2_girder + end2) * 0.5, "9"),
                Node(end2, "s2_bottom"),
            ],
            name=B_STRUT_RIGHT,
        )
        model.add_beam(strut2)

        if self.params.rigid_supports:
            model.support_at_rigid("A1")
            model.support_at_rigid("A2")
        else:
            model.support_at_hinged("A1")
            model.support_at_hinged("A2")

        model.connect_nodes("3", "p1_top")
        model.connect_nodes("5", "p2_top")
        model.connect_nodes("2", "s1_top")
        model.connect_nodes("10", "s1_bottom")
        model.connect_nodes("6", "s2_top")
        model.connect_nodes("11", "s2_bottom")
        return model

    def get_node_uids(self) -> Dict[str, List[str]]:
        uids = super().get_node_uids()
        uids[B_GIRDER] += [
            P_GIRDER_STRUT_LEFT,
            P_GIRDER_PILE_LEFT,
            P_GIRDER_FIELD_CENTER,
            P_GIRDER_PILE_RIGHT,
            P_GIRDER_STRUT_RIGHT,
        ]
        return uids

    def get_support_uids(self) -> List[str]:
        return super().get_support_uids() + [P_LEFT_SUPPORT, P_RIGHT_SUPPORT]
