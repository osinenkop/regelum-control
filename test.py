from rcognita.optimizers import (
    Optimizable,
    partial_positionals,
    casadi_default_config,
    scipy_default_config,
    torch_default_config,
)
import numpy as np
from functools import partial
import torch
from rcognita.models import ModelDeepObjective


bounds = np.array([[-5, 5]])


def f(x, y):
    return (x - 1) ** 2 + y**2 + 10


def constr(x, p1):
    return x**2 - p1


Opt = Optimizable(scipy_default_config)
Opt.register_objective(f)
Opt.register_bounds(bounds)
Opt.recreate_constraints(partial_positionals(constr, {1: 0.5}))
print(Opt.optimize(3, 1))

Opt = Optimizable(casadi_default_config)
x_p = Opt.variable(1)
y_p = Opt.parameter(1)
p1 = Opt.parameter(1)
Opt.register_objective(f, x_p, y_p)
Opt.register_bounds(bounds)
Opt.register_constraint(constr, x_p, p1)
print(Opt.optimize(3, 1, 0.5))


class Child(Optimizable):
    def __init__(self, opt_config) -> None:
        super().__init__(opt_config)
        self.bounds = np.array([[-5, 5]])
        x_p = self.variable(1)
        y_p = self.parameter(1)
        p1 = self.parameter(1)
        self.register_objective(self.f, x_p, y_p)
        self.register_bounds(self.bounds)
        self.register_constraint(self.constr, x_p, p1)

    def f(self, x, y):
        return (x - 1) ** 2 + y**2 + 10

    @staticmethod
    def constr(x, p1):
        return x**2 - p1


model = ModelDeepObjective(2, 3)


class Child(Optimizable):
    def __init__(self, model, opt_config) -> None:
        super().__init__(opt_config)
        self.bounds = np.array([[-5, 5]])
        self.model = model
        self.register_objective(self.f)
        self.register_decision_variable(self.model.parameters())

    def f(self, x):
        return self.model(x)

    @staticmethod
    def constr(x, p1):
        return x**2 - p1


# child = Child(model, torch_default_config)
# print(list(child.model.parameters()))
# child.optimize([1.0, 2.0])
# print(list(child.model.parameters()))
