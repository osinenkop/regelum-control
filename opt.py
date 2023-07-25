import casadi
from scipy.optimize import minimize, NonlinearConstraint


class Optimizable:
    def __init__(self, kind="symbolic", opt_options={}, log_options={}) -> None:
        self.kind = kind
        if self.kind == "symbolic":
            self.__opti = casadi.Opti()
            self.variables = []
            self.opt_func = None
            self.decision_var = None
        self.constraints = []
        self.bounds = []
        self.objective_kind = None
        self.constraints_kind = None
        self.opt_options = opt_options
        self.log_options = log_options

    def register_objective(self, func, *args, kind="symbolic", **kwargs):
        if kind == "symbolic":
            self.register_symbolic_objective(func, *args, **kwargs)
        elif kind == "numeric":
            self.register_numeric_objective(func)

    def register_symbolic_objective(self, func, *args, **kwargs):
        self._objective = func(*args, **kwargs)
        self.__opti.minimize(self._objective)
        self.objective_kind = "symbolic"

    def register_numeric_objective(self, func):
        self._objective = func
        self.objective_kind = "numeric"

    def register_constraint(self, func, *args, kind="symbolic", **kwargs):
        if kind == "symbolic":
            self.register_symbolic_constraint(func, *args, **kwargs)
        elif kind == "numeric":
            self.register_numeric_constraint(func)

    def register_bounds(self, bounds):
        self.bounds = bounds
        if self.kind == "symbolic":
            self.__opti.subject_to(self.__u >= self.decision_variable_bounds[:, 0])
            self.__opti.subject_to(self.__u <= self.decision_variable_bounds[:, 1])

    def register_symbolic_constraint(self, func, *args, **kwargs):
        self.constraints.append(func(*args, **kwargs))

    def register_numeric_constraint(self, func):
        self.constraints.append(func)

    def variable(self, dim):
        var = self.__opti.variable(dim)
        self.variables.append(var)
        return var

    @property
    def defined_params(self):
        return self.variables

    def optimize(self, *args, **kwargs):
        if self.opt_func is None:
            self.opt_func = self.__opti.to_function(
                "min_fun",
                self.variables,
                self.decision_var,
                self.opt_options,
                self.log_options,
            )

        return self.opt_func(*args, **kwargs)
