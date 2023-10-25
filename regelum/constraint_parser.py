from . import RegelumBase
from .__utilities import rc
from abc import abstractmethod, ABC
import numpy as np
from .system import ThreeWheeledRobotNI
from itertools import groupby
from typing import List, Optional
from dataclasses import dataclass


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def row_wise(ignore: Optional[List[str]] = None):
    if ignore is None:
        ignore = []

    def row_wise_inner(constr_func):
        def wrapped_constr(**kwargs):
            if not kwargs:
                return constr_func()
            else:
                lens = [v.shape[0] for k, v in kwargs.items() if k not in ignore]
                assert all_equal(
                    lens
                ), "Numbers of rows must be the same in order to apply row_wise decorator."
                len_single = lens.pop(0)
                return rc.force_column(
                    rc.array(
                        [
                            constr_func(
                                **{
                                    k: v[i, :] if k not in ignore else v
                                    for k, v in kwargs.items()
                                }
                            )
                            for i in range(len_single)
                        ]
                    )
                )

        return wrapped_constr

    return row_wise_inner


class ConstraintParser(RegelumBase, ABC):
    @dataclass
    class ConstraintParameter:
        name: str
        dims: tuple
        data: np.ndarray

    def __init__(self) -> None:
        pass

    def parse_constraints(self, simulation_metadata=None):
        if simulation_metadata is None:
            return {}
        else:
            return self._parse_constraints(simulation_metadata=simulation_metadata)

    def __iter__(self):
        yield from self.constraint_parameters()

    @abstractmethod
    def _parse_constraints(self, simulation_metadata):
        ...

    @abstractmethod
    def constraint_parameters(self):
        ...

    @abstractmethod
    def constraint_function(self, **kwargs):
        ...


class ConstraintParserTrivial(ConstraintParser):
    def _parse_constraints(self, simulation_metadata):
        return {}

    def constraint_parameters(self):
        return []


def assert_shape(array, shape, message):
    assert array.shape == tuple(shape), message


@row_wise(ignore=["state"])
def linear_constraint(weights, bias, state):
    assert state is not None, "state cannot be None"
    return state @ weights + bias


@row_wise(ignore=["state"])
def circle_constraint(coefs, radius, center, state):
    assert state is not None, "state cannot be None"
    return radius**2 - (rc.dot(coefs, (state - center) ** 2))


class ThreeWheeledRobotNIConstantContstraintsParser(ConstraintParser):
    def __init__(
        self, centers=None, coefs=None, radii=None, weights=None, biases=None
    ) -> None:
        self.radii = np.array(radii) if radii is not None else radii
        self.centers = np.array(centers) if centers is not None else centers
        self.coefs = np.array(coefs) if coefs is not None else coefs
        self.weights = np.array(weights) if weights is not None else weights
        self.biases = np.array(biases) if biases is not None else biases

    def _parse_constraints(self, simulation_metadata=None):
        return {"radii": self.radii, "centers": self.centers, "coefs": self.coefs}

    def constraint_parameters(self):
        return [
            self.ConstraintParameter(name, data.shape, data)
            for name, data in [
                ["radii", self.radii],
                ["centers", self.centers],
                ["coefs", self.coefs],
                ["weights", self.weights],
                ["biases", self.biases],
            ]
            if data is not None
        ]

    def constraint_function(
        self,
        weights=None,
        biases=None,
        radii=None,
        centers=None,
        coefs=None,
        state=None,
    ):
        weights = weights if weights is not None else self.weights
        biases = biases if biases is not None else self.biases
        radii = radii if radii is not None else self.radii
        centers = centers if centers is not None else self.centers
        coefs = coefs if coefs is not None else self.coefs

        linear_constraint_values = None
        circle_constraint_values = None
        if weights is not None and biases is not None:
            linear_constraint_values = linear_constraint(
                weights=weights, bias=biases, state=state
            )
        if radii is not None and centers is not None and coefs is not None:
            circle_constraint_values = circle_constraint(
                coefs=coefs, radius=radii, center=centers, state=state
            )
        constraint_values = rc.vstack(
            [
                val
                for val in [linear_constraint_values, circle_constraint_values]
                if val is not None
            ]
        )
        return rc.max(constraint_values)
