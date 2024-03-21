"""Contains a tool box for parsing constraints that are injected outside."""

from . import RegelumBase
from .utils import rg
from abc import abstractmethod, ABC
import numpy as np
from itertools import groupby
from typing import List, Optional, Generator
from dataclasses import dataclass


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def state_wise(func):
    def wrapped_constr(predicted_states=None, **kwargs):
        if len(predicted_states.shape) == 1:
            predicted_states = rg.force_row(predicted_states)
        assert predicted_states is not None, "states cannot be None"
        return rg.max(
            rg.vstack(
                [
                    func(**kwargs, state=predicted_states[i, :])
                    for i in range(predicted_states.shape[0])
                ]
            )
        )

    return wrapped_constr


def row_wise(with_respect_to: Optional[List[str]] = None):
    if with_respect_to is None:
        with_respect_to = []

    def row_wise_inner(constr_func):
        def wrapped_constr(**kwargs):
            if not kwargs:
                return constr_func()
            else:
                lens = [
                    v.shape[0] for k, v in kwargs.items() if k not in with_respect_to
                ]
                assert all_equal(
                    lens
                ), "Numbers of rows must be the same in order to apply row_wise decorator."
                len_single = lens.pop(0)
                return rg.force_column(
                    rg.vstack(
                        [
                            constr_func(
                                **{
                                    k: v[i, :] if k not in with_respect_to else v
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
    """Base class for Constraint Parser."""

    @dataclass
    class ConstraintParameter:
        """Dataclass that represents constraint parameters."""

        name: str
        dims: tuple
        data: np.ndarray

    def __init__(self) -> None:
        """Instantiate ConstraintParser."""
        pass

    def parse_constraints(self, simulation_metadata=None):
        # if simulation_metadata is None:
        #     return {}
        # else:
        return self._parse_constraints(simulation_metadata=simulation_metadata)

    def __iter__(self) -> Generator[ConstraintParameter, None, None]:
        yield from self.constraint_parameters()

    @abstractmethod
    def _parse_constraints(self, simulation_metadata): ...

    @abstractmethod
    def constraint_parameters(self): ...

    @abstractmethod
    def constraint_function(self, **kwargs): ...


class ConstraintParserTrivial(ConstraintParser):
    """Trivial constraint parser that does nothing."""

    def _parse_constraints(self, simulation_metadata):
        return {}

    def constraint_parameters(self):
        return []

    def constraint_function(self, whatever=None):
        return -1


def assert_shape(array, shape, message):
    assert array.shape == tuple(shape), message


@state_wise
@row_wise(with_respect_to=["state"])
def linear_constraint(weights, bias, state):
    assert state is not None, "state cannot be None"
    return state @ rg.array(weights, prototype=state) + rg.array(bias, prototype=state)


@state_wise
@row_wise(with_respect_to=["state"])
def circle_constraint(coefs, radius, center, state):
    assert state is not None, "state cannot be None"
    return rg.array(radius, prototype=state) ** 2 - (
        rg.dot(
            rg.array(coefs, prototype=state),
            (state - rg.array(center, prototype=state)) ** 2,
        )
    )


class CylindricalHalfPlaneConstraintParser(ConstraintParser):
    """Constraint parser for ThreeWheeledRobotKinematic that consist of several circles and lines."""

    def __init__(
        self,
        centers: Optional[np.ndarray] = None,
        coefs: Optional[np.ndarray] = None,
        radii: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        biases: Optional[np.ndarray] = None,
    ) -> None:
        """Instantiate CylindricalHalfPlaneConstraintParser.

        Args:
            centers (Optional[np.ndarray], optional): centers of
                circles, defaults to None
            coefs (Optional[np.ndarray], optional): circle coeficients,
                defaults to None
            radii (Optional[np.ndarray], optional): radii of circles,
                defaults to None
            weights (Optional[np.ndarray], optional): lines'
                coefficients, defaults to None
            biases (Optional[np.ndarray], optional): lines' biases,
                defaults to None
        """
        self.radii = np.array(radii) if radii is not None else radii
        self.centers = np.array(centers) if centers is not None else centers
        self.coefs = np.array(coefs) if coefs is not None else coefs
        self.weights = np.array(weights) if weights is not None else weights
        self.biases = np.array(biases) if biases is not None else biases

    def _parse_constraints(self, simulation_metadata=None):
        return {
            "radii": self.radii,
            "centers": self.centers,
            "coefs": self.coefs,
            "weights": self.weights,
            "biases": self.biases,
        }

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
        predicted_states=None,
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
                weights=weights, bias=biases, predicted_states=predicted_states
            )
        if radii is not None and centers is not None and coefs is not None:
            circle_constraint_values = circle_constraint(
                coefs=coefs,
                radius=radii,
                center=centers,
                predicted_states=predicted_states,
            )
        constraint_values = rg.vstack(
            [
                val
                for val in [linear_constraint_values, circle_constraint_values]
                if val is not None
            ]
        )
        return rg.max(constraint_values)
