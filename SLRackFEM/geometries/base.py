from dataclasses import dataclass
from typing import Dict, List

from ..lib import *
from ..model import CrossSection, Material, Model, BeamProfile


@dataclass
class MaterialParams:
    """Defining parameters of a construction material

    Args:
        E (int): E modulus of the material (kN per square meter)
        G (int): Shear modulus of the material (kN per square meter)
        rho: (int): mass density of the material (kg per m^3)
        fy: (int): elastic limit (kN per square meter)
    """

    E: int
    G: int
    rho: int
    fy: int = 0


@dataclass
class CrossSectionParams:
    """Defining parameters of a cross section

    Args:
        A (float): Cross section area (square meters)
        Iy (float): I-value for bending around Y axis
        Iz (float): I-value for bending around Z axis
        J (float): Torsional constant
    """

    A: float
    Iy: float
    Iz: float
    J: float
    Wy_min: float


@dataclass
class BeamProfileParams:
    material: MaterialParams
    cross_section: CrossSectionParams


class BaseSystem:
    def __init__(self, params, nelem):
        self.params = params
        self.nelem = nelem

    def calc_load_case(self, load_case):
        """Calculate a single load case"""
        model = self.create_model()
        load_case.apply_to_model(model)
        return model.run()

    def create_model(self) -> Model:
        return Model(nelem=self.nelem)

    def add_beam_profile(self, model: Model, params: BeamProfileParams) -> BeamProfile:
        """Add a beam profile specified by given parameters to a given model"""
        m = params.material
        material = Material(m.E * 1000, m.G * 1000, m.rho, m.fy)
        model.add_material(material)
        cs = params.cross_section
        # TODO: what to do with Wy_min?
        cross_section = CrossSection(cs.A, cs.Iy, cs.Iz, cs.J, cs.Wy_min)
        model.add_cross_section(cross_section)
        return BeamProfile(material, cross_section)

    def get_node_uids(self) -> Dict[str, List[str]]:
        """Get a mapping of beam uids present in the model to the node uids on the respective beam"""
        return {}

    def get_support_uids(self) -> List[str]:
        """Get all the uids of nodes which are supports in the FEM model"""
        return []


MAT_S235 = MaterialParams(210000000, 80769200, 7850, 2350000)
CS_W146 = CrossSectionParams(
    8.22 * M2_PER_CM2, 215.51 * M4_PER_CM4, 53.02 * M4_PER_CM4, 268.53 * M4_PER_CM4, Wy_min=1
)
CS_SB_150_62_18 = CrossSectionParams(
    5.11 * M2_PER_CM2, 178.58 * M4_PER_CM4, 22.25 * M4_PER_CM4, 200.83 * M4_PER_CM4, Wy_min=1
)
