import numpy as np
from multiprocessing import Pool
from typing import List

from .model import Model
from .lib import *


class PL:
    """A point load at a specific node"""

    def __init__(self, at, load):
        self.at = at
        self.load = load

    @classmethod
    def new(cls, at, Fx=0.0, Fy=0.0, Fz=0.0, Mx=0.0, My=0.0, Mz=0.0):
        return cls(at, [Fx, Fy, Fz, Mx, My, Mz])

    def __mul__(self, other):
        return type(self)(self.at, [i * other for i in self.load])

    __rmul__ = __mul__


class DL:
    """A distributed load along a specific beam"""

    def __init__(self, load, start=None, to=None):
        self.load = load
        self.start = start
        self.to = to

    @classmethod
    def new(cls, Fx=0.0, Fy=0.0, Fz=0.0, Mx=0.0, My=0.0, Mz=0.0, start=None, to=None):
        return cls([Fx, Fy, Fz, Mx, My, Mz], start=start, to=to)

    def __mul__(self, other):
        return type(self)([i * other for i in self.load], start=self.start, to=self.to)

    __rmul__ = __mul__


class LoadCollection:
    """A collection of point or distributed loads ('Load case')"""

    def __init__(self, loads={}, dead_weight=False, factor=1.0):
        self.loads = loads
        self.dead_weight = dead_weight
        self.factor = factor

    def apply_to_model(self, model: Model):
        if self.dead_weight:
            for beam in model.beams:
                # kg/m^3 * m^2 * N/kg = N/m
                gpm = beam.material.rho * beam.cross_section.A * 9.81 * self.factor
                model.add_distributed_load(beam, [0, 0, gpm, 0, 0, 0])
        if isinstance(self.loads, dict):
            for beam, loads in self.loads.items():
                for load in loads:
                    load = load * self.factor
                    if isinstance(load, PL):
                        model.add_point_load(beam, load.at, load.load)
                    elif isinstance(load, DL):
                        model.add_distributed_load(beam, load.load, start=load.start, to=load.to)
        elif isinstance(self.loads, list):
            for load in self.loads:
                (self.factor * load).apply_to_model(model)

    def __mul__(self, other):
        return type(self)(self.loads, self.dead_weight, factor=self.factor * other)

    __rmul__ = __mul__


class LoadCombination:
    """A combination of different load cases"""

    def __init__(self, load_cases, factor=1.0):
        self.load_cases = [factor * load_case for load_case in load_cases]

    def __mul__(self, other):
        return type(self)(self.load_cases, factor=other)

    __rmul__ = __mul__


class LoadCombinationResults:
    """Holds the results for all load cases within a load combination

    Also maximal (positive as well as negative) values for displacements/internal forces are stored.
    """

    def __init__(self, framat_results, results):
        self.results = results
        self.abm = framat_results.get("mesh").get("abm")
        minima = self.minima = {}
        maxima = self.maxima = {}
        temporary = {
            "local_displacements": [r.local_displacements for r in results],
            "global_displacements": [r.global_displacements for r in results],
            "local_forces": [r.local_forces for r in results],
            "global_forces": [r.global_forces for r in results],
            "reaction_forces": [r.reaction_forces for r in results],
        }
        for key, values in temporary.items():
            out_min = minima[key] = np.array(values[0], copy=True)
            out_max = maxima[key] = np.array(values[0], copy=True)
            for a in values[1:]:
                np.minimum(out_min, a, out=out_min)
                np.maximum(out_max, a, out=out_max)

    def get_local_displacements(self, node, max_otherwise_min):
        dest = self.maxima if max_otherwise_min else self.minima
        return self.abm.gnv(dest["local_displacements"], node)

    def get_global_displacements(self, node, max_otherwise_min):
        dest = self.maxima if max_otherwise_min else self.minima
        return self.abm.gnv(dest["global_displacements"], node)

    def get_local_forces(self, node, max_otherwise_min):
        dest = self.maxima if max_otherwise_min else self.minima
        return self.abm.gnv(dest["local_forces"], node)

    def get_global_forces(self, node, max_otherwise_min):
        dest = self.maxima if max_otherwise_min else self.minima
        return self.abm.gnv(dest["global_forces"], node)


# calculate a single load combination, helper for parallel processing
def calc_lc(model, lc):
    return model.calc_load_case(lc)[1]


def calc_load_combinations(
    model, *load_combinations, worker_processes=1
) -> List[LoadCombinationResults]:
    """Calculate multiple load combinations/load cases in parallel with multiprocessing"""
    if worker_processes == 0:
        raise NotImplementedError()
    with Pool(worker_processes) as p:
        load_cases = [
            lc for load_combination in load_combinations for lc in load_combination.load_cases
        ]
        results = p.starmap_async(calc_lc, [(model, lc) for lc in load_cases[1:]])
        framat_results, res_lc0 = model.calc_load_case(load_cases[0])
        results_complete = [res_lc0, *results.get()]
        lc_results = []
        for load_combination in load_combinations:
            lc_results.append(
                LoadCombinationResults(
                    framat_results, results_complete[: len(load_combination.load_cases)]
                )
            )
            results_complete = results_complete[len(load_combination.load_cases) :]
        return lc_results
