import numpy as np
from framat import Model as FramatModel
from framat._element import Element

from .lib import Vector

_UID_COUNTER = 0


class ModelPart:
    """Base model part identified by an UID"""

    def __init__(self, uid=None):
        global _UID_COUNTER
        if not uid:
            self.uid = f"uid{_UID_COUNTER}"
            _UID_COUNTER = _UID_COUNTER + 1
        else:
            self.uid = uid


class Material(ModelPart):
    def __init__(self, E, G, rho, fy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.E = E
        self.G = G
        self.rho = rho
        self.fy = fy


class CrossSection(ModelPart):
    def __init__(self, A, Iy, Iz, J, Wy_min, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.Wy_min = Wy_min


class BeamProfile:
    """A beam profile consisting of a material and a cross section"""

    def __init__(self, material: Material, cross_section: CrossSection):
        self.material = material
        self.cross_section = cross_section

    def get_N_el_d(self):
        return self.material.fy * self.cross_section.A

    def get_M_el_d(self):
        return self.material.fy * self.cross_section.Wy_min


class Node(ModelPart):
    """A single FEM node"""

    def __init__(self, v: Vector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coords = v.coords


class Beam:
    """A beam in the FEM model"""

    def __init__(self, beam_profile, nodes, up_direction=None, nelem=None, name=None):
        self.material = beam_profile.material
        self.cross_section = beam_profile.cross_section
        self.nodes = nodes
        self.up_direction = up_direction
        self.nelem = nelem
        self.name = name


class Model:
    """Wrapper class around the framat Model class

    Provides convenient methods to manipulate the FEM model.
    """

    def __init__(self, nelem=50):
        self.nelem = nelem
        self.model = FramatModel()
        self._bc = self.model.set_feature("bc")
        self._pp = self.model.set_feature("post_proc")
        self._beams = {}
        self._beam_names = {}

    def add_material(self, material: Material):
        try:
            if self.model.get("material", uid=material.uid) is not None:
                return
        except KeyError:
            pass
        mat = self.model.add_feature("material", uid=material.uid)
        mat.set("E", material.E)
        mat.set("G", material.G)
        mat.set("rho", material.rho)

    def add_cross_section(self, cross_section: CrossSection):
        try:
            if self.model.get("cross_section", uid=cross_section.uid) is not None:
                return
        except KeyError:
            pass
        cs = self.model.add_feature("cross_section", uid=cross_section.uid)
        cs.set("A", cross_section.A)
        cs.set("Iy", cross_section.Iy)
        cs.set("Iz", cross_section.Iz)
        cs.set("J", cross_section.J)

    def add_beam(self, beam: Beam):
        p = self.model.add_feature("beam")
        self._beams[beam] = p
        if beam.name:
            self._beam_names[beam.name] = beam
        for n in beam.nodes:
            p.add("node", n.coords, n.uid)
        p.set("nelem", beam.nelem if beam.nelem else self.nelem)
        first_node = beam.nodes[0]
        last_node = beam.nodes[-1]
        p.add("material", {"from": first_node.uid, "to": last_node.uid, "uid": beam.material.uid})
        p.add(
            "cross_section",
            {"from": first_node.uid, "to": last_node.uid, "uid": beam.cross_section.uid},
        )
        # Assume straight beam and up direction in XZ plane, if not given (2D problem)
        if not beam.up_direction:
            direction = [l[1] - l[0] for l in zip(beam.nodes[0].coords, beam.nodes[-1].coords)]
            if direction[1] != 0:
                raise ValueError("Beam not in XZ plane and no up direction vector given")
            beam.up_direction = [direction[2], 0, -direction[0]]
        p.add("orientation", {"from": first_node.uid, "to": last_node.uid, "up": beam.up_direction})

    def beam(self, name):
        """Get the Beam object for a given name"""
        if isinstance(name, Beam):
            return name
        return self._beam_names[name]

    @property
    def beams(self):
        return self._beams.keys()

    def connect_nodes(self, node1, node2, dof=["ux", "uy", "uz"]):
        """Connect two FEM nodes typically belonging to different beams"""
        self._bc.add(
            "connect",
            {
                "node1": node1 if type(node1) is str else node1.uid,
                "node2": node2 if type(node2) is str else node2.uid,
                "fix": dof,
            },
        )

    def support_at(self, node, dof):
        self._bc.add("fix", {"node": node if type(node) is str else node.uid, "fix": dof})

    def support_at_hinged(self, node):
        """Add a hinged support at the given node"""
        self.support_at(node, dof=["ux", "uy", "uz", "thx", "thz"])

    def support_at_rigid(self, node):
        """Add a rigid support at the given node"""
        self.support_at(node, dof=["all"])

    # Load specific
    def add_point_load(self, beam, at, load):
        """Add a point load (in N) at the given node"""
        feature = self._beams[self.beam(beam)]
        feature.add("point_load", {"at": at, "load": load})

    def add_distributed_load(self, beam, load, start=None, to=None):
        """Add a distributed load (in N/m) along a given beam

        Optionally, a start and end node for the load can be specified
        """
        beam = self.beam(beam)
        if not start:
            start = beam.nodes[0].uid
        if not to:
            to = beam.nodes[-1].uid

        self._beams[beam].add("distr_load", {"from": start, "to": to, "load": load})

    def run(self):
        """Run the FEM calculation and return the results

        Returns a tuple consisting of the generated Framat model and the Results object"""
        results = self.model.run()
        return results, Results.create(results)


class Results:
    """Results for a given FEM calculation

    Only force/displacement vectors are stored, no model/mesh data.
    """

    def __init__(self, gd, ld, gf, lf, rf):
        self.global_displacements = gd
        self.local_displacements = ld
        self.global_forces = gf
        self.local_forces = lf
        self.reaction_forces = rf

    @classmethod
    def create(cls, results):
        abm = results.get("mesh").get("abm")
        displacements = results.get("tensors").get("U")
        mult = None
        displacements = np.reshape(displacements, (-1, 6))
        local_displacements = np.empty(displacements.shape)
        local_forces = np.zeros(displacements.shape)
        used = 0
        for i in range(len(abm.beams)):
            beam = abm.beams[i]
            # assume same direction along complete beam line
            sub_beam_values = list(beam.values())
            elem = Element.from_abstract_element(sub_beam_values[0])

            # global to local system transformation matrix
            mult = elem.T[0:6, 0:6]
            beam_displacements = abm.gbv(displacements, i)
            sub_beams = len(beam_displacements)
            local_beam_displacements = mult @ beam_displacements.T
            local_displacements[used : used + sub_beams] = local_beam_displacements.T
            for nr, s in enumerate(sub_beam_values):
                elem = Element.from_abstract_element(s)
                sml = elem.stiffness_matrix_local
                # f = k * u
                local_forces[used + nr : used + nr + 2] = np.reshape(
                    sml @ np.reshape(local_displacements[used + nr : used + nr + 2], (12, 1)),
                    (2, 6),
                )
            used += len(beam_displacements)

        k = results.get("tensors").get("K")
        displacements = results.get("tensors").get("U")
        f_internal = k * displacements
        f_internal = np.reshape(f_internal, (-1, 6))
        displacements = np.reshape(displacements, (-1, 6))

        return cls(
            displacements,
            local_displacements,
            f_internal,
            local_forces,
            results.get("tensors").get("F_react"),
        )

    def get_elem_len(self, elem):
        return np.linalg.norm(elem.p2.coord - elem.p1.coord)

    def __reduce__(self):
        # Needs to be defined so that an instance can be recreated when returned from multiprocessing
        return Results, (
            self.global_displacements,
            self.local_displacements,
            self.global_forces,
            self.local_forces,
            self.reaction_forces,
        )
