"""
This module contains optimization routines to be used in optimal controllers, actors, critics etc.

"""

from rcognita.__utilities import rc
import scipy as sp
from scipy.optimize import minimize
import numpy as np
import warnings

try:
    from casadi import vertcat, nlpsol, DM, MX, Function

except (ModuleNotFoundError, ImportError):
    warnings.warn_explicit(
        "\nImporting casadi failed. You may still use rcognita, but"
        + " without symbolic optimization capability. ",
        UserWarning,
        __file__,
        42,
    )

from abc import ABC, abstractmethod
import time

try:
    import torch.optim as optim
    import torch

except ModuleNotFoundError:
    warnings.warn_explicit(
        "\nImporting Torch failed. You may still use rcognita, but"
        + " without PyTorch. ",
        UserWarning,
        __file__,
        42,
    )


class Optimizer(ABC):
    """
    Abstract base class for optimizers.
    """

    @property
    @abstractmethod
    def engine(self):
        """Name of the optimization engine being used"""
        return "engine_name"

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @staticmethod
    def verbose(opt_func):
        """
        A static method decorator that makes the decorated function verbose.

        This method will print the optimization time of the decorated function
        if the `verbose` attribute of the instance is set to True.

        Parameters:
        opt_func (function): The function to be decorated.

        Returns:
        function: The decorated function.
        """

        def wrapper(self, *args, **kwargs):
            tic = time.time()
            result = opt_func(self, *args, **kwargs)
            toc = time.time()
            if self.verbose:
                print(f"result optimization time:{toc-tic} \n")

            return result

        return wrapper


class SciPyOptimizer(Optimizer):
    """
    Optimizer class using the SciPy optimization library.

    Attributes:
        engine (str): Name of the optimization engine.
        opt_method (str): Optimization method to use.
        opt_options (dict): Options for the optimization method.
        verbose (bool): Whether to print optimization progress and timing.
    """

    engine = "SciPy"

    def __init__(self, opt_method, opt_options, verbose=False):
        """
        Initialize a SciPyOptimizer instance.

        :param opt_method: str, the name of the optimization method to use.
        :param opt_options: dict, options for the optimization method.
        :param verbose: bool, whether to print the optimization time and the objective function value before and after optimization.
        """
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.verbose = verbose

    @Optimizer.verbose
    def optimize(self, objective, initial_guess, bounds, constraints=(), verbose=False):
        """
        Optimize the objective function using the specified method and options.

        :param objective: function, the objective function to optimize.
        :param initial_guess: array-like, the initial guess for the optimization.
        :param bounds: tuple, the lower and upper bounds for the optimization.
        :param constraints: tuple, the equality and inequality constraints for the optimization.
        :param verbose: bool, whether to print the objective function value before and after optimization.
        :return: array-like, the optimal solution.
        """

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


class CasADiOptimizer(Optimizer):
    engine = "CasADi"

    def __init__(self, opt_method, opt_options, verbose=False):
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.verbose = verbose
        """
        Initialize the CasADiOptimizer object.
        
        :param opt_method: The optimization method to use (string).
        :type opt_method: str
        :param opt_options: A dictionary of options for the optimization method.
        :type opt_options: dict
        :param verbose: Whether or not to print messages during optimization (default: False).
        :type verbose: bool, optional
        """

    @Optimizer.verbose
    def optimize(
        self,
        objective,
        initial_guess,
        bounds,
        constraints=(),
        decision_variable_symbolic=None,
    ):
        """
        Optimize the given objective function using the CasADi optimization engine.
        
        :param objective: The objective function to optimize.
        :type objective: function
        :param initial_guess: The initial guess for the optimization variables.
        :type initial_guess: numpy array
        :param bounds: A tuple of lower and upper bounds for the optimization variables.
        :type bounds: tuple
        :param constraints: Any constraints to enforce during optimization (default: no constraints).
        :type constraints: tuple, optional
        :param decision_variable_symbolic: A list of symbolic variables representing the optimization variables.
        :type decision_variable_symbolic: list
        :return: The optimized decision variables.
        :rtype: numpy array
        """
        optimization_problem = {
            "f": objective,
            "x": vertcat(decision_variable_symbolic),
            "g": vertcat(*constraints),
        }

        atol = 1e-10
        if isinstance(constraints, (tuple, list)):
            upper_bound_constraint = [atol for _ in constraints]
        elif isinstance(constraints, (MX, DM, int, float)):
            upper_bound_constraint = [atol]

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

        return rc.to_np_1D(result["x"])


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

    @Optimizer.verbose
    def optimize(self, initial_guess, *args):
        """
        Optimize the given objective function using the CasADi optimization engine.
        
        :param objective: The objective function to optimize.
        :type objective: function
        :param initial_guess: The initial guess for the optimization variables.
        :type initial_guess: numpy array
        :param bounds: A tuple of lower and upper bounds for the optimization variables.
        :type bounds: tuple
        :param constraints: Any constraints to enforce during optimization (default: no constraints).
        :type constraints: tuple, optional
        :param decision_variable_symbolic: A list of symbolic variables representing the optimization variables.
        :type decision_variable_symbolic: list
        :return: The optimized decision variables.
        :rtype: numpy array
        """
        for _ in range(self.N_steps):
            initial_guess = self.grad_step(initial_guess, *args)

        return initial_guess


class TorchOptimizer(Optimizer):
    """
    Optimizer class that uses PyTorch as its optimization engine.
    """

    engine = "Torch"

    def __init__(self, opt_options, iterations=1, opt_method=None, verbose=False):
        """
        Initialize an instance of TorchOptimizer.
        
        :param opt_options: Options for the PyTorch optimizer.
        :type opt_options: dict
        :param iterations: Number of iterations to optimize the model.
        :type iterations: int
        :param opt_method: PyTorch optimizer class to use. If not provided, Adam is used.
        :type opt_method: torch.optim.Optimizer
        :param verbose: Whether to print optimization progress.
        :type verbose: bool
        """
        if opt_method is None:
            opt_method = torch.optim.Adam
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.iterations = iterations
        self.verbose = verbose
        self.loss_history = []

    @Optimizer.verbose
    def optimize(self, objective, model, model_input):
        """
        Optimize the model with the given objective.
        
        :param objective: Objective function to optimize.
        :type objective: callable
        :param model: Model to optimize.
        :type model: torch.nn.Module
        :param model_input: Inputs to the model.
        :type model_input: torch.Tensor
        """
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


class BruteForceOptimizer(Optimizer):
    """
    Optimizer that searches for the optimal solution by evaluating all possible variants in parallel."
    """

    engine = "bruteforce"

    def __init__(self, possible_variants, N_parallel_processes=0):
        """
        Initialize an instance of BruteForceOptimizer.

        :param N_parallel_processes: number of processes to use in parallel
        :type N_parallel_processes: int
        :param possible_variants: list of possible variants to evaluate
        :type possible_variants: list
        """
        self.N_parallel_processes = N_parallel_processes
        self.possible_variants = possible_variants

    def element_wise_maximization(self, x):
        """
        Find the variant that maximizes the reward for a given element.
        
        :param x: element to optimize
        :type x: tuple
        :return: variant that maximizes the reward
        :rtype: int
        """
        reward_function = lambda variant: self.objective(variant, x)
        reward_function = np.vectorize(reward_function)
        values = reward_function(self.possible_variants)
        return self.possible_variants[np.argmax(values)]

    def optimize(self, objective, weights):
        """
        Maximize the objective function over the possible variants.
        
        :param objective: The objective function to maximize.
        :type objective: Callable
        :param weights: The weights to optimize.
        :type weights: np.ndarray
        :return: The optimized weights.
        :rtype: np.ndarray
        """
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
