"""
This module contains optimization routines to be used in optimal controllers, policies, critics etc.

"""
import rcognita.base
from rcognita.__utilities import rc
import scipy as sp
from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
import warnings

try:
    from casadi import vertcat, nlpsol, DM, MX, Function, Opti

except (ModuleNotFoundError, ImportError):
    pass

from abc import ABC, abstractmethod
import time

try:
    import torch.optim as optim
    import torch
    from torch.utils.data import Dataset, DataLoader
    from rcognita.data_buffers import UpdatableSampler

except ModuleNotFoundError:
    from unittest.mock import MagicMock

    torch = MagicMock()
    UpdatableSampler = MagicMock()

from rcognita.callbacks import apply_callbacks
from typing import Callable, Optional, Union
from functools import partial


class Optimizer:
    pass


class LazyOptimizer:
    _engine_name = "LazyOptimizer"

    def __init__(
        self,
        class_object: Optional[type] = None,
        objective_function: Optional[Callable] = None,
        opt_method=None,
        opt_options: Optional[dict] = None,
        print_options: Optional[dict] = None,
        decision_variable=None,
        decision_variable_bounds: Optional[np.ndarray] = None,
        decision_variable_dim: Optional[list] = None,
        free_parameters: Optional[dict[str, int]] = None,
        constraints: Union[
            list[Optional[Callable]], tuple[Optional[Callable]], None
        ] = None,
        is_instantiated=False,
    ) -> None:
        if not is_instantiated and class_object is None:
            raise ValueError("class_object must be specified if not instantiated")
        assert objective_function is None or callable(
            objective_function
        ), "objective_function must be callable"
        assert opt_options is None or isinstance(
            opt_options, dict
        ), "opt_options must be of type dict"
        assert print_options is None or isinstance(
            print_options, dict
        ), "print_options must be of type dict"
        assert constraints is None or all(
            [callable(c) for c in constraints]
        ), "constraints must be callable"

        self.class_object = class_object
        if decision_variable_dim is not None:
            self.specify_decision_variable_dimensions(
                decision_variable_dim=decision_variable_dim
            )
        if free_parameters is not None:
            self.specify_parameters(free_parameters=free_parameters)

        self.specify_decision_variable(decision_variable)
        self.specify_parameters(free_parameters)
        if objective_function is not None:
            self.minimize(objective_function)
        else:
            self.objective_function = None

        self.opt_options = opt_options if opt_options is not None else {}
        self.print_options = print_options if print_options is not None else {}
        self.specify_opt_method(opt_method)

        self.decision_variable_bounds = decision_variable_bounds
        self.subject_to(constraints)
        if self.decision_variable_bounds is not None:
            self.apply_bounds(decision_variable_bounds)
        else:
            self.decision_variable_bounds = None

    def minimize(self, objective_function: Callable) -> None:
        assert callable(objective_function), "objective_function must be callable"
        self.objective_function = objective_function

    def subject_to(
        self,
        constraints: Union[
            list[Optional[Callable]], tuple[Optional[Callable]], None
        ] = None,
    ) -> None:
        if constraints is None:
            ## TODO: change to log level info
            print("No constraints specified")
        self.constraints = constraints

    def specify_opt_method(self, opt_method) -> None:
        self.opt_method = opt_method

    def specify_decision_variable_dimensions(
        self,
        decision_variable_dim: Optional[list] = None,
    ) -> None:
        if decision_variable_dim is not None:
            self.decision_variable_dim = decision_variable_dim
        else:
            print("No decision variable dimension specified")

    def apply_bounds(
        self, decision_variable_bounds: Optional[np.ndarray] = None
    ) -> None:
        assert (
            isinstance(decision_variable_bounds, np.ndarray)
            and len(decision_variable_bounds.shape) == 2
        ), "decision_variable_bounds must be a 2D np.ndarray"
        self.decision_variable_bounds = decision_variable_bounds

    def specify_parameters(self, free_parameters: Optional[dict] = None) -> None:
        self.free_parameters = free_parameters

    def specify_decision_variable(self, decision_variable) -> None:
        self.decision_variable = decision_variable

    def instantiate(self):
        assert self.class_object is not None
        return self.class_object(
            objective_function=self.objective_function,
            opt_method=self.opt_method,
            opt_options=self.opt_options,
            print_options=self.print_options,
            decision_variable=self.decision_variable,
            decision_variable_bounds=self.decision_variable_bounds,
            decision_variable_dim=self.decision_variable_dim,
            free_parameters=self.free_parameters,
            constraints=self.constraints,
            is_instantiated=True,
        )

    def optimize(self, initial_guess, **free_parameters):
        pass


class SciPyOptimizer(LazyOptimizer):
    def subject_to(
        self,
        constraints: Union[
            list[Optional[Callable]], tuple[Optional[Callable]], None
        ] = None,
    ) -> None:
        if constraints:
            constraints = [
                NonlinearConstraint(c, -np.inf, 0.0) for c in constraints if c
            ]
        super().subject_to(constraints)

    def optimize(self, initial_guess, *free_parameters, **free_parameters_kwargs):
        if self.decision_variable_bounds is not None:
            bounds = sp.optimize.Bounds(
                self.decision_variable_bounds[:, 0],
                self.decision_variable_bounds[:, 1],
                keep_feasible=True,
            )
        else:
            bounds = None
        opt_result = minimize(
            partial(
                self.objective_function, *free_parameters, **free_parameters_kwargs
            ),
            x0=initial_guess,
            method=self.opt_method,
            bounds=bounds,
            options=self.opt_options,
            constraints=self.constraints,
            tol=1e-7,
        )

        return opt_result.x


class CasADiOptimizer(LazyOptimizer):
    _engine = "CasADi"

    def __init__(self, *args, **kwargs):
        self.__opti = Opti()
        self.mx_constraints = []
        self.mx_parameters = []
        self.minimizer = None
        super().__init__(*args, **kwargs)
        self.minimizer = self.__opti.to_function("minimize", [self.__u], [self.__u])

    def specify_decision_variable_dimensions(
        self, decision_variable_dim: Optional[list] = None
    ) -> None:
        super().specify_decision_variable_dimensions(decision_variable_dim)
        assert (
            self.decision_variable_dim is not None
        ), "decision_variable_dim must be specified"
        self.__u = self.__opti.variable(self.decision_variable_dim)

    def specify_parameters(self, free_parameters: Optional[dict] = None) -> None:
        super().specify_parameters(free_parameters)
        if self.free_parameters:
            for k, v in self.free_parameters:
                self.mx_parameters.append(self.__opti.parameter(v))

    def apply_bounds(
        self, decision_variable_bounds: Optional[np.ndarray] = None
    ) -> None:
        super().apply_bounds(decision_variable_bounds)
        if self.decision_variable_bounds is not None:
            self.__opti.subject_to(self.__u >= self.decision_variable_bounds[:, 0])
            self.__opti.subject_to(self.__u <= self.decision_variable_bounds[:, 1])

    def specify_opt_method(self, opt_method) -> None:
        super().specify_opt_method(opt_method)
        assert self.opt_method is not None, "opt_method must be specified"
        self.__opti.solver(self.opt_method, self.opt_options, self.print_options)

    def subject_to(
        self,
        constraints: Union[
            list[Optional[Callable]], tuple[Optional[Callable]], None
        ] = None,
    ) -> None:
        super().subject_to(constraints)
        if self.constraints:
            for c in self.constraints:
                if c:
                    self.__opti.subject_to(c(self.__u) <= 0)

    def minimize(self, objective_function: Callable) -> None:
        super().minimize(objective_function)
        assert (
            self.objective_function is not None
        ), "objective_function must be specified"
        self.__opti.minimize(objective_function(self.__u))

    def optimize(
        self,
        initial_guess,
    ):
        result = self.minimizer(initial_guess)
        return result


class TorchOptimizer(LazyOptimizer):
    """
    Optimizer class that uses PyTorch as its optimization engine.
    """

    _engine = "Torch"

    def __init__(
        self, opt_options, model, iterations=1, opt_method=None, verbose=False
    ):
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
        self.model = model
        self.optimizer = self.opt_method(model.parameters(), **self.opt_options)

    def optimize(
        self, objective, model_input=None
    ):  # remove model and add parameters instead
        """
        Optimize the model with the given objective.

        :param objective: Objective function to optimize.
        :type objective: callable
        :param model: Model to optimize.
        :type model: torch.nn.Module
        :param model_input: Inputs to the model.
        :type model_input: torch.Tensor
        """

        for _ in range(self.iterations):
            self.optimizer.zero_grad()
            loss = objective(model_input)
            loss.backward()
            self.optimizer.step()


class TorchDataloaderOptimizer(LazyOptimizer):
    """
    Optimizer class that uses PyTorch as its optimization engine.
    """

    engine = "Torch"

    def __init__(
        self,
        opt_options,
        model,
        shuffle=True,
        opt_method=None,
        batch_size=None,
        sheduler_method=None,
        sheduler_options=None,
        batch_sampler=None,
        verbose=False,
    ):
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
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.model = model
        self.optimizer = self.opt_method(self.model.parameters(), **self.opt_options)
        self.sheduler_method = sheduler_method
        self.sheduler_options = sheduler_options
        self.sheduler = self.sheduler_method(self.optimizer, **self.sheduler_options)

        if isinstance(batch_sampler, UpdatableSampler):
            self.batch_sampler = batch_sampler
        else:
            self.batch_sampler = None

    @apply_callbacks()
    def post_epoch(self, idx_epoch, last_epoch_objective):
        return idx_epoch, last_epoch_objective

    def optimize(self, objective, dataset):  # remove model and add parameters instead
        """
        Optimize the model with the given objective.

        :param objective: Objective function to optimize.
        :type objective: callable
        :param model: Model to optimize.
        :type model: torch.nn.Module
        :param model_input: Inputs to the model.
        :type model_input: torch.Tensor
        """

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size if self.batch_size is not None else len(dataset),
        )

        batch_sample = next(iter(dataloader))
        self.optimizer.zero_grad()
        objective_value = objective(batch_sample)
        last_epoch_objective = objective_value.item()
        objective_value.backward()
        self.optimizer.step()
        self.sheduler.step()

        self.post_epoch(1, last_epoch_objective)


class TorchProjectiveOptimizer(LazyOptimizer):
    """
    Optimizer class that uses PyTorch as its optimization engine.
    """

    engine = "Torch"

    def __init__(
        self,
        bounds,
        opt_options,
        prediction_horizon=0,
        iterations=1,
        opt_method=None,
        verbose=False,
    ):
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
        self.bounds = bounds
        if opt_method is None:
            opt_method = torch.optim.Adam
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.iterations = iterations
        self.verbose = verbose
        self.loss_history = []
        self.action_size = self.bounds[:, 1].shape[0]
        self.upper_bound = torch.squeeze(
            torch.tile(torch.tensor(self.bounds[:, 1]), (1, prediction_horizon + 1))
        )

    def optimize(self, *model_input, objective, model):
        """
        Optimize the model with the given objective.

        :param objective: Objective function to optimize.
        :type objective: callable
        :param model: Model to optimize.
        :type model: torch.nn.Module
        :param model_input: Inputs to the model.
        :type model_input: torch.Tensor
        """
        optimizer = self.opt_method([model_input[0]], **self.opt_options)
        # optimizer.zero_grad()

        for _ in range(self.iterations):
            optimizer.zero_grad()
            loss = objective(*model_input)
            # loss_before = loss.detach().numpy()
            loss.backward()
            optimizer.step()
            for param in [model_input[0]]:
                param.requires_grad = False
                param /= self.upper_bound
                param.clamp_(-1, 1)
                param *= self.upper_bound
                param.requires_grad = True
            # optimizer.zero_grad()
            # loss_after = objective(*model_input).detach().numpy()
            # print(loss_before - loss_after)
            if self.verbose:
                print(objective(*model_input))
        # self.loss_history.append([loss_before, loss_after])
        model.weights = torch.nn.Parameter(model_input[0][: self.action_size])
        return model_input[0]


class BruteForceOptimizer(LazyOptimizer):
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
