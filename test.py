from rcognita.optimizers import (
    Optimizable,
    partial_positionals,
    casadi_default_config,
    scipy_default_config,
)
import numpy as np
from functools import partial


def f(x, y):
    return (x - 1) ** 2 + y**2 + 10


bounds = np.array([[-5, 5]])


def c1(x, p1):
    return x**2 - p1


O = Optimizable(scipy_default_config)
O.register_objective(f)
O.register_bounds(bounds)
O.recreate_constraints(partial_positionals(c1, {1: 0.5}))
print(O.optimize(3, 1))

O = Optimizable(casadi_default_config)
x_p = O.variable(1)
y_p = O.parameter(1)
p1 = O.parameter(1)
O.register_objective(f, x_p, y_p)
O.register_bounds(bounds)
O.register_constraint(c1, x_p, p1)
print(O.optimize(3, 1, 0.5))
