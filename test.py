from rcognita.optimizers import (
    Optimizable,
    casadi_default_config,
)
import numpy as np


class Example(Optimizable):
    def __init__(self, opt_config):
        super().__init__(opt_config)
        self.x = self.create_variable(1, 1, name="x")
        self.y = self.create_variable(1, 1, name="y")
        self.register_objective(self.f, [self.x, self.y])
        self.register_bounds(self.x, np.array([[-5, 5]]))

    @staticmethod
    def f(x, y):
        return x + y

    @staticmethod
    def h(x, y):
        return x**2 + y**2 - 9


e = Example(casadi_default_config)
print(e.objectives, e.objectives[0].placeholders)
e.fix_variables(["x"])
print(e.objectives, e.objectives[0].placeholders)
