from rcognita.optimizers import (
    VarContainer,
    Variable,
    FunctionWithSignature,
    FuncContainer,
    Optimizable,
    casadi_default_config,
)


class Example(Optimizable):
    def __init__(self, opt_config):
        super().__init__(opt_config)
        self.x = self.create_variable(1, 1, name="x")
        self.y = self.create_variable(1, 1, name="y")
        self.register_objective(self.f, [self.x, self.y])

    @staticmethod
    def f(x, y):
        return x + y


e = Example(casadi_default_config)
