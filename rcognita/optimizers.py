"""
This module contains optimization routines to be used in optimal controllers, actors, critics etc.

"""

from rcognita.utilities import rc
import scipy as sp
from scipy.optimize import minimize
import numpy as np
import warnings

try:
    from casadi import vertcat, nlpsol, DM, SX, Function

except ModuleNotFoundError:
    warnings.warn_explicit(
        "\nImporting casadi failed. You may still use rcognita, but"
        + " without symbolic optimization capability. ",
        UserWarning,
        __file__,
        42,
    )

from abc import ABC, abstractmethod
import time
import torch.optim as optim
import torch
from multiprocessing import Pool
import matplotlib.pyplot as plt


class BaseOptimizer(ABC):
    """
    Optimizer blueprint.

    """

    @property
    @abstractmethod
    def engine(self):
        return "engine_name"

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    def verbose(opt_func):
        def wrapper(self, *args, **kwargs):
            tic = time.time()
            result = opt_func(self, *args, **kwargs)
            toc = time.time()
            if self.verbose:
                print(f"result optimization time:{toc-tic} \n")

            return result

        return wrapper


class SciPyOptimizer(BaseOptimizer):
    engine = "SciPy"

    def __init__(self, opt_method, opt_options, verbose=False):
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.verbose = verbose

    @BaseOptimizer.verbose
    def optimize(self, objective, initial_guess, bounds, constraints=(), verbose=False):

        weight_bounds = sp.optimize.Bounds(bounds[0], bounds[1], keep_feasible=True)

        before_opt = objective(initial_guess)
        opt_result = minimize(
            objective,
            x0=initial_guess,
            method=self.opt_method,
            bounds=weight_bounds,
            options=self.opt_options,
            constraints=constraints,
            tol=1e-7,
        )
        if verbose:
            print(f"before:{before_opt},\nafter:{opt_result.fun}")

        return opt_result.x


class CasADiOptimizer(BaseOptimizer):
    engine = "CasADi"

    def __init__(self, opt_method, opt_options, verbose=False):
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.verbose = verbose

    @BaseOptimizer.verbose
    def optimize(
        self,
        objective,
        initial_guess,
        bounds,
        constraints=(),
        decision_variable_symbolic=None,
    ):
        optimization_problem = {
            "f": objective,
            "x": vertcat(decision_variable_symbolic),
            "g": vertcat(*constraints),
        }

        if isinstance(constraints, tuple):
            upper_bound_constraint = [0 for _ in constraints]
        elif isinstance(constraints, (SX, DM, int, float)):
            upper_bound_constraint = [0]

        try:
            solver = nlpsol(
                "solver", self.opt_method, optimization_problem, self.opt_options,
            )
        except Exception as e:
            print(e)
            return initial_guess

        if upper_bound_constraint is not None and len(upper_bound_constraint) > 0:
            result = solver(
                x0=initial_guess,
                lbx=bounds[0],
                ubx=bounds[1],
                ubg=upper_bound_constraint,
            )
        else:
            result = solver(x0=initial_guess, lbx=bounds[0], ubx=bounds[1])

        ##### DEBUG
        # g1 = Function("g1", [symbolic_var], [constraints])

        # print(g1(result["x"]))
        ##### DEBUG

        return result["x"]


class GradientOptimizer(CasADiOptimizer):
    def __init__(
        self,
        objective,
        learning_rate,
        N_steps,
        grad_norm_upper_bound=1e-2,
        verbose=False,
    ):
        self.objective = objective
        self.learning_rate = learning_rate
        self.N_steps = N_steps
        self.grad_norm_upper_bound = grad_norm_upper_bound
        self.verbose = verbose

    def substitute_args(self, initial_guess, *args):
        cost_function, symbolic_var = rc.function2MX(
            self.objective, initial_guess=initial_guess, force=True, *args
        )

        return cost_function, symbolic_var

    def grad_step(self, initial_guess, *args):
        cost_function, symbolic_var = self.substitute_args(initial_guess, *args)
        cost_function = Function("f", [symbolic_var], [cost_function])
        gradient = rc.autograd(cost_function, symbolic_var)
        grad_eval = gradient(initial_guess)
        norm_grad = rc.norm_2(grad_eval)
        if norm_grad > self.grad_norm_upper_bound:
            grad_eval = grad_eval / norm_grad * self.grad_norm_upper_bound

        initial_guess_res = initial_guess - self.learning_rate * grad_eval
        return initial_guess_res

    @BaseOptimizer.verbose
    def optimize(self, initial_guess, *args):
        for _ in range(self.N_steps):
            initial_guess = self.grad_step(initial_guess, *args)

        return initial_guess


class TorchOptimizer(BaseOptimizer):
    engine = "Torch"

    def __init__(
        self, opt_options, iterations=1, opt_method=torch.optim.Adam, verbose=False
    ):
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.iterations = iterations
        self.verbose = verbose
        self.loss_history = []

    @BaseOptimizer.verbose
    def optimize(self, objective, model, model_input):
        optimizer = self.opt_method(
            model.parameters(), **self.opt_options, weight_decay=0
        )
        # optimizer.zero_grad()

        for _ in range(self.iterations):
            optimizer.zero_grad()
            loss = objective(model_input)
            loss_before = loss.detach().numpy()
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()
            loss_after = objective(model_input).detach().numpy()
            print(loss_before - loss_after)
            if self.verbose:
                print(objective(model_input))
        self.loss_history.append([loss_before, loss_after])


class BruteForceOptimizer(BaseOptimizer):
    engine = "Parallel"

    def __init__(self, N_parallel_processes, possible_variants):
        self.N_parallel_processes = N_parallel_processes
        self.possible_variants = possible_variants

    def element_wise_maximization(self, x):
        reward_function = lambda variant: self.objective(variant, x)
        reward_function = np.vectorize(reward_function)
        values = reward_function(self.possible_variants)
        return self.possible_variants[np.argmax(values)]

    def optimize(self, objective, weights):
        self.weights = weights
        self.objective = objective
        indices = tuple(
            [(i, j) for i in range(weights.shape[0]) for j in range(weights.shape[1])]
        )
        for x in indices:
            self.weights[x] = self.element_wise_maximization(x)

        return self.weights

        # with Pool(self.n_pools) as p:
        #     result_weights = p.map(
        #         self.element_wise_maximization,
        #         np.nditer(self.weights, flags=["external_loop"]),
        #     )[0]
        # return result_weights
