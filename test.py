from rcognita.optimizers import (
    Optimizable,
    OptimizerConfig,
    casadi_default_config,
    torch_default_config,
    scipy_default_config,
)
import numpy as np


class Example(Optimizable):
    def __init__(self, optimizer_config: OptimizerConfig) -> None:
        Optimizable.__init__(self, optimizer_config)
        self.bounds = np.array([[-2, 2]])

    @staticmethod
    def f(x, y=1, z=2):
        return (x - 3) ** 2 + y**2

    @staticmethod
    def g(x, y):
        return x**2 + y**2 - 9

    @staticmethod
    def h(x, p):
        return -(x**2)

    def define_problem(self):
        x = self.create_variable(1, 1, name="x")
        y = self.create_variable(1, 1, name="y", is_constant=True)
        z = self.create_variable(1, 1, name="z", is_constant=True)
        p = self.create_variable(1, 1, name="p", is_constant=True)
        self.register_objective(self.f, [x, y, z])
        self.register_bounds(variable_to_bound=x, bounds=self.bounds)
        self.register_constraint(self.g, [x, y])
        self.register_constraint(self.h, [x, p])


e = Example(scipy_default_config)
print(e.optimize(x=-1, y=0, z=1, p=1, raw=False))

e = Example(casadi_default_config)
print(e.optimize(x=-1, y=0, z=1, p=1, raw=False))
