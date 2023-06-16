"""
This module contains model classes.
These can be used in system dynamics fitting, critic and other tasks

Updates to come.

"""
import numpy as np
import os, sys

import rcognita.base

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

from __utilities import rc, rej_sampling_rvs, torch_safe_log
import numpy as np
import warnings

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.distributions.multivariate_normal import MultivariateNormal
except ModuleNotFoundError:
    from unittest.mock import MagicMock

    torch = MagicMock()
    nn = MagicMock()
    F = MagicMock()
    MultivariateNormal = MagicMock()

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import OrderedDict


def force_positive_def(func):
    def positive_def_wrapper(self, *args, **kwargs):
        if self.force_positive_def:
            return rc.soft_abs(func(self, *args, **kwargs))
        else:
            return func(self, *args, **kwargs)

    return positive_def_wrapper


class Model(rcognita.base.RcognitaBase, ABC):
    """
    Blueprint of a model.
    """

    def __call__(self, *args, weights=None, use_stored_weights=False):
        if use_stored_weights is False:
            if weights is not None:
                return self.forward(*args, weights=weights)
            else:
                return self.forward(*args, weights=self.weights)
        else:
            return self.cache.forward(*args, weights=self.cache.weights)

    @property
    @abstractmethod
    def model_name(self):
        return "model_name"

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    def update_weights(self, weights):
        self.weights = weights

    def cache_weights(self, weights=None):
        if "cache" not in self.__dict__.keys():
            self.cache = deepcopy(self)

        if weights is None:
            self.cache.update_weights(self.weights)
        else:
            self.cache.update_weights(weights)

    def update_and_cache_weights(self, weights):
        self.cache_weights(weights)
        self.update_weights(weights)

    def restore_weights(self):
        """
        Assign the weights of the cached model to the active model.
        This may be needed when pytorch optimizer resulted in unsatisfactory weights, for instance.

        """

        self.update_and_cache_weights(self.cache.weights)


class ModelSS:
    model_name = "state-space"
    """
    State-space model
            
    .. math::
        \\begin{array}{ll}
			\\hat x^+ & = A \\hat x + B u, \\newline
			y^+  & = C \\hat x + D u.
        \\end{array}                 
        
    Attributes
    ---------- 
    A, B, C, D : : arrays of proper shape
        State-space model parameters.
    initial_guessset : : array
        Initial state estimate.
            
    """

    def __init__(self, A, B, C, D, initial_guessest):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.initial_guessest = initial_guessest

    def update_pars(self, Anew, Bnew, Cnew, Dnew):
        self.A = Anew
        self.B = Bnew
        self.C = Cnew
        self.D = Dnew

    def updateIC(self, initial_guesssetNew):
        self.initial_guessset = initial_guesssetNew


class ModelQuadLin(Model):
    """
    Quadratic-linear model.

    """

    model_name = "quad-lin"

    def __init__(
        self,
        dim_input,
        single_weight_min=1e-6,
        single_weight_max=1e2,
        force_positive_def=True,
    ):
        self.dim_weights = int((dim_input + 1) * dim_input / 2 + dim_input)
        self.weight_min = single_weight_min * rc.ones(self.dim_weights)
        self.weight_max = single_weight_max * rc.ones(self.dim_weights)
        self.weights_init = (self.weight_min + self.weight_max) / 20.0
        self.weights = self.weights_init
        self.force_positive_def = force_positive_def
        self.update_and_cache_weights(self.weights)

    @force_positive_def
    def forward(self, *argin, weights=None):
        if len(argin) > 1:
            vec = rc.concatenate(tuple(argin))
        else:
            vec = argin[0]

        polynom = rc.uptria2vec(rc.outer(vec, vec))
        polynom = rc.concatenate([polynom, vec])
        result = rc.dot(weights, polynom)

        return result


class ModelQuadratic(Model):
    """
    Quadratic model. May contain mixed terms.

    """

    model_name = "quadratic"

    def __init__(
        self,
        dim_input,
        single_weight_min=1e-6,
        single_weight_max=1e2,
        force_positive_def=True,
    ):
        self.dim_weights = int((dim_input + 1) * dim_input / 2)
        self.weight_min = single_weight_min * rc.ones(self.dim_weights)
        self.weight_max = single_weight_max * rc.ones(self.dim_weights)
        self.weights_init = (self.weight_min + self.weight_max) / 2.0
        self.weights = self.weights_init
        self.force_positive_def = force_positive_def
        self.update_and_cache_weights(self.weights)

    @force_positive_def
    def forward(self, *argin, weights=None):
        if len(argin) > 1:
            vec = rc.concatenate(tuple(argin))
        else:
            vec = argin[0]

        if isinstance(vec, tuple):
            vec = vec[0]

        polynom = rc.uptria2vec(rc.outer(vec, vec))
        result = rc.dot(weights, polynom)

        return result


class ModelQuadraticSquared(ModelQuadratic):
    """
    Quadratic model. May contain mixed terms.

    """

    model_name = "quadratic-squared"

    def forward(self, *argin, weights=None):
        result = super().forward(*argin, weights=weights)

        result = result**2 / 1e5

        return result


class ModelQuadNoMix(Model):
    """
    Quadratic model (no mixed terms).

    """

    model_name = "quad-nomix"

    def __init__(
        self,
        dim_input,
        single_weight_min=1e-6,
        single_weight_max=1e3,
    ):
        self.dim_weights = dim_input
        self.weight_min = single_weight_min * rc.ones(self.dim_weights)
        self.weight_max = single_weight_max * rc.ones(self.dim_weights)
        self.weights_init = (self.weight_min + self.weight_max) / 2.0
        self.weights = self.weights_init
        self.force_positive_def = force_positive_def
        self.update_and_cache_weights(self.weights)

    def forward(self, *argin, weights=None):
        if len(argin) > 1:
            vec = rc.concatenate(tuple(argin))
        else:
            vec = argin[0]

        if isinstance(vec, tuple):
            vec = vec[0]

        polynom = vec * vec

        result = rc.dot(weights, polynom)

        return result


class ModelQuadNoMix2D(Model):
    """
    Quadratic model (no mixed terms).

    """

    model_name = "quad-nomix"

    def __init__(
        self,
        dim_input,
        single_weight_min=1e-6,
        single_weight_max=1e2,
    ):
        self.dim_weights = dim_input
        self.weight_min = single_weight_min * rc.ones(self.dim_weights)[:2]
        self.weight_max = single_weight_max * rc.ones(self.dim_weights)[:2]
        self.weights_init = (self.weight_min + self.weight_max) / 2.0
        self.weights = self.weights_init
        self.force_positive_def = force_positive_def
        self.update_and_cache_weights(self.weights)

    def forward(self, *argin, weights=None):
        if len(argin) > 1:
            vec = rc.concatenate(tuple(argin))
        else:
            vec = argin[0]

        if isinstance(vec, tuple):
            vec = vec[0]

        polynom = vec[:2] * vec[:2]

        result = rc.dot(weights, polynom)

        return result


class ModelWeightContainer(Model):
    """
    Trivial model, which is typically used in actor in which actions are being optimized directly.

    """

    model_name = "action-sequence"

    def __init__(self, dim_output, weights_init=None):
        self.dim_output = dim_output
        self.weights = weights_init
        self.weights_init = weights_init
        self.update_and_cache_weights(self.weights)

    def forward(self, *argin, weights=None):
        return weights[: self.dim_output]


class ModelQuadMix(Model):
    model_name = "quad-mix"

    def __init__(self, dim_input, weight_min=1.0, weight_max=1e3):
        self.dim_weights = int(
            self.dim_output + self.dim_output * self.dim_input + self.dim_input
        )
        self.weight_min = weight_min * np.ones(self.dim_weights)
        self.weight_max = weight_max * np.ones(self.dim_weights)

    def _forward(self, vec, weights):
        v1 = rc.force_column(v1)
        v2 = rc.force_column(v2)

        polynom = rc.concatenate([v1**2, rc.kron(v1, v2), v2**2])
        result = rc.dot(weights, polynom)

        return result


class ModelQuadForm(Model):
    """
    Quadratic form.

    """

    model_name = "quad_form"

    def __init__(self, weights=None):
        self.weights = weights

    def forward(self, *argin, weights=None):
        if len(argin) != 2:
            raise ValueError("ModelQuadForm assumes two vector arguments!")

        vec = rc.concatenate(tuple(argin))

        try:
            result = vec.T @ weights @ vec
        except RuntimeError:
            result = vec.T @ torch.tensor(weights, requires_grad=False).double() @ vec

        result = rc.squeeze(result)

        return result


class ModelBiquadForm(Model):
    """
    Bi-quadratic form.

    """

    model_name = "biquad_form"

    def __init__(self, weights):
        self.weights = weights

    def forward(self, *argin, weights=None):
        if len(argin) != 2:
            raise ValueError("ModelBiquadForm assumes two vector arguments!")

        vec = rc.concatenate(tuple(argin))

        result = vec.T**2 @ weights[0] @ vec**2 + vec.T @ weights[1] @ vec

        result = rc.squeeze(result)

        return result


class ModelNN(nn.Module):
    """
    Class of pytorch neural network models. This class is not to be used barebones.
    Instead, you should inherit from it and specify your concrete architecture.

    """

    model_name = "NN"

    def __call__(self, *args, weights=None, use_stored_weights=False):
        if use_stored_weights is False:
            if weights is not None:
                return self.forward(*args, weights=weights)
            else:
                return self.forward(*args)
        else:
            return self.cache.forward(*args)

    @property
    def cache(self):
        """
        Isolate parameters of cached model from the current model
        """
        return self.cached_model[0]

    def detach_weights(self):
        """
        Excludes the model's weights from the pytorch computation graph.
        This is needed to exclude the weights from the decision variables in optimization problems.
        An example is temporal-difference optimization, where the old critic is to be treated as a frozen model.

        """
        for variable in self.parameters():
            variable.detach_()

    def cache_weights(self, whatever=None):
        """
        Assign the active model weights to the cached model followed by a detach.

        This method also backs up itself and performs this operation only once upon the initialization procedure
        """
        if "cached_model" not in self.__dict__.keys():
            self.cached_model = (
                deepcopy(self),
            )  ## this is needed to prevent cached_model's parameters to be parsed by model init hooks

        self.cache.load_state_dict(self.weights)
        self.cache.detach_weights()

    @property
    def weights(self):
        return self.state_dict()

    def update_weights(self, whatever=None):
        pass

    def weights2dict(self, weights_to_parse):
        """
        Transform weights as a numpy array into a dictionary compatible with pytorch.

        """

        weights_to_parse = torch.tensor(weights_to_parse)

        new_state_dict = {}

        length_old = 0

        for param_tensor in self.state_dict():
            weights_size = self.state_dict()[param_tensor].size()
            weights_length = math.prod(self.state_dict()[param_tensor].size())
            new_state_dict[param_tensor] = torch.reshape(
                weights_to_parse[length_old : length_old + weights_length],
                tuple(weights_size),
            )
            length_old = weights_length

        return new_state_dict

    def update_and_cache_weights(self, weights=None):
        if weights is not None:
            for item in weights:
                weights[item].requires_grad_()
            weights = self.load_state_dict(weights)
        # self.load_state_dict(self.weights2dict(weights))
        self.cache_weights()

    def restore_weights(self):
        """
        Assign the weights of the cached model to the active model.
        This may be needed when pytorch optimizer resulted in unsatisfactory weights, for instance.

        """

        self.update_and_cache_weights(self.cache.state_dict())

    def soft_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (Torch model): weights will be copied from
            target_model (Torch model): weights will be copied to
            tau (float): interpolation parameter

        """
        for target_param, local_param in zip(
            self.cache.parameters(), self.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def __call__(self, *argin, weights=None, use_stored_weights=False):
        if len(argin) > 1:
            argin = rc.concatenate(argin)
        else:
            argin = argin[0]

        argin = argin if isinstance(argin, torch.Tensor) else torch.tensor(argin)

        if use_stored_weights is False:
            if weights is not None:
                result = self.forward(argin, weights)
            else:
                result = self.forward(argin)
        else:
            result = self.cache.forward(argin)

        return result


class ModelQuadNoMixTorch(ModelNN):
    """
    pytorch neural network of one layer: fully connected.

    """

    def __init__(
        self,
        dim_observation,
        dim_action,
        dim_hidden=20,
        weights=None,
        force_positive_def=False,
    ):
        super().__init__()

        # self.fc1 = nn.Linear(dim_observation + dim_action, 1, bias=False)
        self.w1 = torch.nn.Parameter(
            torch.ones(dim_observation + dim_action, requires_grad=True)
        )

        if weights is not None:
            self.load_state_dict(weights)

        self.double()
        self.cache_weights()
        self.force_positive_def = force_positive_def

    def forward(self, input_tensor, weights=None):
        if weights is not None:
            self.update(weights)

        x = input_tensor**2
        x = self.w1**2 @ x

        return x


class ModelDDQNAdvantage(ModelNN):
    def __init__(
        self,
        dim_observation,
        dim_action,
        dim_hidden=40,
    ):
        super().__init__()

        self.fc1 = nn.Linear(dim_observation + dim_action, dim_hidden)
        self.a1 = nn.ReLU()
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.a2 = nn.ReLU()
        self.fc3 = nn.Linear(dim_hidden, dim_hidden)
        self.a3 = nn.ReLU()
        self.fc4 = nn.Linear(dim_hidden, 1)

        self.double()

    def forward(self, input_tensor):
        x = input_tensor
        x = self.fc1(x)
        x = self.a1(x)
        x = self.fc2(x)
        x = self.a2(x)
        x = self.fc3(x)
        x = self.a3(x)
        x = self.fc4(x)

        return torch.squeeze(x)


class ModelDeepObjective(ModelNN):
    def __init__(
        self,
        dim_observation,
        dim_hidden=40,
    ):
        super().__init__()

        self.fc1 = nn.Linear(dim_observation, dim_hidden)
        self.a1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.a2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(dim_hidden, dim_hidden)
        self.a3 = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(dim_hidden, 1)

        self.double()
        self.cache_weights()

    def forward(self, input_tensor):
        x = input_tensor
        x = self.fc1(x)
        x = self.a1(x)
        x = self.fc2(x)
        x = self.a2(x)
        x = self.fc3(x)
        x = self.a3(x)
        x = self.fc4(x)

        return torch.squeeze(x)


class ModelDDQN(ModelNN):
    def __init__(
        self,
        dim_observation,
        dim_action,
        actions_grid,
        dim_hidden=40,
    ):
        super().__init__()

        self.dim_observation = dim_observation
        self.actions_grid = actions_grid
        self.critic = ModelDeepObjective(
            dim_observation=dim_observation,
            dim_hidden=dim_hidden,
        )
        self.advantage = ModelDDQNAdvantage(
            dim_observation=dim_observation,
            dim_action=dim_action,
            dim_hidden=dim_hidden,
        )

        self.double()

        self.cache_weights()

    def forward(self, input_tensor, weights=None):
        observation_action, observation = (
            input_tensor,
            input_tensor[: self.dim_observation],
        )

        objective = self.critic(observation)
        advantage = self.advantage(observation_action)

        advantage_grid_mean = sum(
            [
                self.advantage(
                    torch.cat(
                        [observation, torch.tensor(action_grid_item).double()], dim=0
                    )
                )
                for action_grid_item in self.actions_grid.T
            ]
        ) / len(self.actions_grid)

        return objective + (advantage - advantage_grid_mean)


class ModelDQNSimple(ModelNN):
    def __init__(
        self,
        dim_observation,
        dim_action,
        dim_hidden=40,
        weights=None,
        force_positive_def=False,
        bias=False,
        leaky_relu_coef=0.2,
    ):
        super().__init__()

        self.in_layer = nn.Linear(dim_observation + dim_action, dim_hidden, bias=bias)
        self.hidden1 = nn.Linear(dim_hidden, dim_hidden, bias=bias)
        self.hidden2 = nn.Linear(dim_hidden, dim_hidden, bias=bias)
        self.out_layer = nn.Linear(dim_hidden, 1, bias=bias)
        self.leaky_relu_coef = leaky_relu_coef
        self.force_positive_def = force_positive_def

        self.double()

        if weights is not None:
            self.load_state_dict(weights)

        self.cache_weights()

    @force_positive_def
    def forward(self, input_tensor, weights=None):
        if weights is not None:
            self.update(weights)

        x = input_tensor
        x = self.in_layer(x)
        x = nn.LeakyReLU(self.leaky_relu_coef)(x)
        x = self.hidden1(x)
        x = nn.LeakyReLU(self.leaky_relu_coef)(x)
        x = self.hidden2(x)
        x = nn.LeakyReLU(self.leaky_relu_coef)(x)
        x = self.out_layer(x)

        return torch.squeeze(x)


class ModelPerceptronCalf(Model):
    model_name = "DQN_simple_casadi"
    weights_dict = {}

    @property
    def weights(self):
        return rc.concatenate(
            [
                rc.reshape_CasADi_as_np(w["weights"], (w["dim_in"] * w["dim_out"], 1))
                for w in self.weights_dict.values()
            ]
        )

    @weights.setter
    def weights(self, weights):
        self.weights_dict = self.wrap_weights_to_dict(weights)

    def get_weights_markup(self):
        weights_markup = {}
        idx = 0
        for key, weights_meta in self.weights_dict.items():
            weights_markup[key] = {
                "from_idx": idx,
                "to_idx": idx + weights_meta["dim_in"] * weights_meta["dim_out"],
            }
            idx += weights_meta["dim_in"] * weights_meta["dim_out"]
        return weights_markup

    def wrap_weights_to_dict(self, weights):
        if isinstance(weights, np.ndarray):
            weights = rc.array(weights, rc_type=rc.CASADI)

        weights_markup = self.get_weights_markup()
        weights_dict = {}
        for key, weights_meta in weights_markup.items():
            weights_dict[key] = {
                "weights": rc.reshape_CasADi_as_np(
                    weights[weights_meta["from_idx"] : weights_meta["to_idx"]],
                    (
                        self.weights_dict[key]["dim_in"],
                        self.weights_dict[key]["dim_out"],
                    ),
                ),
                "dim_in": self.weights_dict[key]["dim_in"],
                "dim_out": self.weights_dict[key]["dim_out"],
            }
        return weights_dict

    class CasadiLayerLinear:
        def __init__(self, weights, name, bias=None):
            self.weights = weights
            self.bias = bias
            self.name = name

        def forward(self, argin, weights):
            argin = (
                rc.array(argin, prototype=weights["weights"])
                if isinstance(argin, np.ndarray)
                else argin
            )

            if self.bias is not None:
                argin = rc.vstack(
                    (argin, rc.ones((1, rc.shape(argin)[1]), prototype=argin))
                )

            return (argin.T @ weights["weights"]).T

        def __call__(self, argin, weights=None):
            if weights is None:
                weights = self.weights

            return self.forward(argin, weights=weights)

    class LeakyReLU:
        def __init__(self, leaky_relu_coef=0.01):
            self.leaky_relu_coef = leaky_relu_coef

        def __call__(self, x):
            return rc.LeakyReLU(x, negative_slope=self.leaky_relu_coef)

    def register_linear_weights(self, weights, name, dim_in, dim_out):
        self.weights_dict[name] = {
            "weights": weights,
            "dim_in": dim_in,
            "dim_out": dim_out,
        }

    def Linear(self, dim_in, dim_out, name, bias=None):
        """
        Here we take bias into account by introducing an additional row in the weights matrix.
        It is equivalent to $xW + b$
        """
        if bias is not None:
            dim_in = dim_in + 1
        weights = np.random.uniform(
            self.weight_min, self.weight_max, size=(dim_in, dim_out)
        )
        weights = rc.array(weights, rc_type=rc.CASADI)

        self.register_linear_weights(weights, name, dim_in, dim_out)
        return self.CasadiLayerLinear(weights=weights, name=name, bias=bias)

    def __init__(
        self,
        dim_observation,
        dim_action,
        single_weight_min=-1.0,
        single_weight_max=1.0,
        dim_hidden=40,
        force_positive_def=False,
        bias=False,
        leaky_relu_coef=0.2,
    ):

        self.weight_min = single_weight_min
        self.weight_max = single_weight_max
        self.in_layer = self.Linear(
            dim_observation + dim_action, dim_hidden, name="in_layer", bias=bias
        )
        self.hidden1 = self.Linear(dim_hidden, dim_hidden, bias=bias, name="hidden1")
        self.hidden2 = self.Linear(dim_hidden, dim_hidden, bias=bias, name="hidden2")
        self.out_layer = self.Linear(dim_hidden, 1, bias=bias, name="out_layer")
        self.leaky_relu = self.LeakyReLU(leaky_relu_coef)
        self.leaky_relu_coef = leaky_relu_coef
        self.force_positive_def = force_positive_def

        self.update_and_cache_weights(self.weights)

    @force_positive_def
    def forward(self, *argin, weights=None):
        if len(argin) > 1:
            vec = rc.concatenate(tuple(argin))
        else:
            vec = argin[0]

        weights_dict = self.weights_dict
        if weights is not None:
            weights_dict = self.wrap_weights_to_dict(weights)

        x = vec
        x = self.in_layer(x, weights_dict[self.in_layer.name])
        x = self.leaky_relu(x)
        x = self.hidden1(x, weights_dict[self.hidden1.name])
        x = self.leaky_relu(x)
        x = self.hidden2(x, weights_dict[self.hidden2.name])
        x = self.leaky_relu(x)
        x = self.out_layer(x, weights_dict[self.out_layer.name])

        return x


class ModelDQN(ModelNN):
    """
    pytorch neural network DQN

    """

    def __init__(
        self,
        dim_observation,
        dim_action,
        dim_hidden=40,
        weights=None,
        force_positive_def=False,
        bias=False,
        leaky_relu_coef=0.2,
    ):
        super().__init__()

        self.in_layer = nn.Linear(dim_observation + dim_action, dim_hidden, bias=bias)
        self.hidden1 = nn.Linear(dim_hidden, dim_hidden, bias=bias)
        self.hidden2 = nn.Linear(dim_hidden, dim_hidden, bias=bias)
        self.hidden3 = nn.Linear(dim_hidden, dim_hidden, bias=bias)
        self.hidden4 = nn.Linear(dim_hidden, dim_hidden, bias=bias)
        self.hidden5 = nn.Linear(dim_hidden, dim_hidden, bias=bias)
        self.out_layer = nn.Linear(dim_hidden, 1, bias=bias)
        self.leaky_relu_coef = leaky_relu_coef
        self.force_positive_def = force_positive_def

        self.double()

        if weights is not None:
            self.load_state_dict(weights)

        self.cache_weights()

    @force_positive_def
    def forward(self, input_tensor, weights=None):
        if weights is not None:
            self.update(weights)

        x = input_tensor
        x = self.in_layer(x)
        x = nn.LeakyReLU(self.leaky_relu_coef)(x)
        x = self.hidden1(x)
        x = nn.LeakyReLU(self.leaky_relu_coef)(x)
        x = self.hidden2(x)
        x = nn.LeakyReLU(self.leaky_relu_coef)(x)
        x = self.hidden3(x)
        x = nn.LeakyReLU(self.leaky_relu_coef)(x)
        x = self.hidden4(x)
        x = nn.LeakyReLU(self.leaky_relu_coef)(x)
        x = self.hidden5(x)
        x = nn.LeakyReLU(self.leaky_relu_coef)(x)
        x = self.out_layer(x)

        return torch.squeeze(x)


class ModelWeightContainerTorch(ModelNN):
    """
    Pytorch weight container for actor

    """

    def __init__(self, action_init):
        super().__init__()

        self.p = torch.nn.Parameter(torch.tensor(action_init, requires_grad=True))

        self.double()
        self.force_positive_def = False
        self.cache_weights()

    def forward(self, observation):
        return self.weights


class LookupTable(Model):
    model_name = "lookup-table"

    def __init__(self, *dims):
        dims = tuple(
            rc.concatenate(tuple([rc.atleast_1d(dim) for dim in dims])).astype(int)
        )
        self.weights = rc.zeros(dims)
        self.update_and_cache_weights(self.weights)

    def __call__(self, *argin, use_stored_weights=False):
        if use_stored_weights is False:
            result = self.forward(*argin)
        else:
            result = self.cache.forward(*argin)
        return result

    def forward(self, *argin, weights=None):
        indices = tuple(
            rc.squeeze(
                rc.concatenate(tuple([rc.atleast_1d(rc.array(ind)) for ind in argin]))
            ).astype(int)
        )
        return self.weights[indices]


class WeightClipper:
    def __init__(self, weight_min=None, weight_max=None):
        self.weight_min = weight_min
        self.weight_max = weight_max

    def __call__(self, module):
        if self.weight_min is not None or self.weight_max is not None:
            # filter the variables to get the ones you want
            w = module.weight.data
            w = w.clamp(self.weight_min, self.weight_max)
            module.weight.data = w


class ModelFc(ModelNN):
    def __init__(
        self,
        dim_observation,
        dim_action,
        use_derivative=False,
        weight_min=None,
        weight_max=None,
    ):
        super().__init__()
        self.weight_min = weight_min
        self.weight_max = weight_max

        if use_derivative:
            dim_observation = dim_observation * 2

        self.in_layer = nn.Linear(dim_observation, dim_action, bias=False)

        self.double()
        self.force_positive_def = False
        self.cache_weights()

    def forward(self, input_tensor, weights=None):
        if weights is not None:
            self.update(weights)

        self.in_layer.apply(WeightClipper(self.weight_min, self.weight_max))

        x = input_tensor
        x = self.in_layer(x)

        return x


class ModelNNElementWiseProduct(ModelNN):
    def __init__(
        self, dim_observation, weight_min=None, weight_max=None, use_derivative=False
    ):

        super().__init__()

        if use_derivative:
            dim_observation = dim_observation * 2

        self.dim_observation = dim_observation

        self.register_parameter(
            name="dot_layer",
            param=torch.nn.Parameter(
                0.1 * torch.ones(self.dim_observation),
                requires_grad=True,
            ),
        )
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.cache_weights()

    def forward(self, input_tensor):
        dot_layer = dict(self.named_parameters())["dot_layer"]
        dot_layer = dot_layer.clamp(min=self.weight_min, max=self.weight_max)
        if len(input_tensor.shape) == 2:
            return input_tensor * dot_layer[None, :]
        return input_tensor * dot_layer


def weights_init_normal(model, std):
    """Takes in a module and initializes all linear layers with weight
    values taken from a normal distribution."""

    classname = model.__class__.__name__
    # for every Linear layer in a model
    if classname.find("Linear") != -1:
        y = model.in_features
        # m.weight.data shoud be taken from a normal distribution
        model.weight.data.normal_(0.0, std)
        # m.bias.data should be 0
        model.bias.data.fill_(0)


class GaussianPDFModel(ModelNN):
    def __init__(
        self,
        dim_observation,
        dim_action,
        dim_hidden,
        std,
        action_bounds,
        leakyrelu_coef=0.2,
        normalize_output_coef=400.0,
    ):
        super().__init__()

        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden
        self.leakyrelu_coef = leakyrelu_coef
        self.std = std
        self.normalize_output_coef = normalize_output_coef

        self.perceptron = nn.Sequential(
            nn.Linear(self.dim_observation, self.dim_hidden),
            nn.LeakyReLU(self.leakyrelu_coef),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.LeakyReLU(self.leakyrelu_coef),
            nn.Linear(self.dim_hidden, dim_action),
        )

        self.register_parameter(
            name="scale_tril_matrix",
            param=torch.nn.Parameter(
                (self.std * torch.eye(dim_action)).float(),
                requires_grad=False,
            ),
        )

        self.register_parameter(
            name="action_bounds",
            param=torch.nn.Parameter(
                torch.tensor(action_bounds).float(),
                requires_grad=False,
            ),
        )

        self.cache_weights()

    def get_unscale_coefs_from_minus_one_one_to_action_bounds(self):
        action_bounds = self.get_parameter("action_bounds")
        unscale_bias, unscale_multiplier = (
            action_bounds.mean(dim=1),
            (action_bounds[:, 1] - self.action_bounds[:, 0]) / 2.0,
        )
        return unscale_bias, unscale_multiplier

    def unscale_from_minus_one_one_to_action_bounds(self, x):
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()

        return x * unscale_multiplier + unscale_bias

    def scale_from_action_bounds_to_minus_one_one(self, y):
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()

        return (y - unscale_bias) / unscale_multiplier

    def get_means(self, observation):
        assert 1 - 3 * self.std > 0, "1 - 3 std should be greater than 0"
        # We should guarantee with good probability that sampled actions are within action bounds that are scaled to [-1, 1]
        return (1 - 3 * self.std) * torch.tanh(
            self.perceptron(observation) / self.normalize_output_coef
        )

    def split_tensor_to_observations_actions(self, observations_actions_tensor):
        if len(observations_actions_tensor.shape) == 1:
            observation, action = (
                observations_actions_tensor[: self.dim_observation],
                observations_actions_tensor[self.dim_observation :],
            )
        elif len(observations_actions_tensor.shape) == 2:
            observation, action = (
                observations_actions_tensor[:, : self.dim_observation],
                observations_actions_tensor[:, self.dim_observation :],
            )
        else:
            raise ValueError("Input tensor has unexpected dims")

        return observation, action

    def log_probs(self, observations_actions, weights=None):
        if weights is not None:
            self.update(weights)

        observations, actions = self.split_tensor_to_observations_actions(
            observations_actions
        )
        means = self.get_means(observations)
        scaled_actions = self.scale_from_action_bounds_to_minus_one_one(actions)

        return MultivariateNormal(
            loc=means,
            scale_tril=self.get_parameter("scale_tril_matrix"),
        ).log_prob(scaled_actions)

    def sample(self, observation):
        mean = self.get_means(observation)
        sampled_scaled_action = MultivariateNormal(
            loc=mean,
            scale_tril=self.get_parameter("scale_tril_matrix"),
        ).sample()

        sampled_action = self.unscale_from_minus_one_one_to_action_bounds(
            sampled_scaled_action
        )

        return sampled_action


class TanhGaussianPDFModel(ModelNN):
    def __init__(
        self,
        dim_observation,
        dim_action,
        scaled_std,
        use_derivative=False,
        weight_min=None,
        weight_max=None,
        action_bounds=None,
        safe_log_eps=1e-10,
    ):
        super().__init__()

        if use_derivative:
            dim_observation = dim_observation * 2

        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.scaled_std = scaled_std
        self.safe_log_eps = safe_log_eps

        self.weight_clipper = WeightClipper(weight_min, weight_max)
        self.in_layer = nn.Linear(self.dim_observation, dim_action, bias=False)
        self.in_layer.apply(self.weight_clipper)

        self.register_parameter(
            name="action_bounds",
            param=torch.nn.Parameter(
                torch.tensor(action_bounds).float(),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            name="cov_matrix",
            param=torch.nn.Parameter(
                torch.diag(self.scaled_std**2 * torch.ones(dim_action).float()),
                requires_grad=False,
            ),
        )
        self.cache_weights()

    def get_unscale_coefs_from_minus_one_one_to_action_bounds(self):
        action_bounds = self.get_parameter("action_bounds")
        unscale_bias, unscale_multiplier = (
            action_bounds.mean(dim=1),
            (action_bounds[:, 1] - self.action_bounds[:, 0]) / 2.0,
        )
        return unscale_bias, unscale_multiplier

    def unscale_from_minus_one_one_to_action_bounds(self, x):
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()

        return x * unscale_multiplier + unscale_bias

    def scale_from_action_bounds_to_minus_one_one(self, y):
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()

        return (y - unscale_bias) / unscale_multiplier

    def forward_observation(self, observation):
        self.in_layer.apply(self.weight_clipper)
        x = self.in_layer(observation)
        out = torch.tanh(x)
        return out

    def split_tensor_to_observation_action(self, input_tensor):
        if len(input_tensor.shape) == 1:
            observation, action = (
                input_tensor[: self.dim_observation],
                input_tensor[self.dim_observation :],
            )
        elif len(input_tensor.shape) == 2:
            observation, action = (
                input_tensor[:, : self.dim_observation],
                input_tensor[:, self.dim_observation :],
            )
        else:
            raise ValueError("Input tensor has unexpected dims")

        return observation, action

    def forward(self, input_tensor, weights=None):
        if weights is not None:
            self.update(weights)

        observation, action = self.split_tensor_to_observation_action(input_tensor)
        mean = self.forward_observation(observation)
        scaled_action = self.scale_from_action_bounds_to_minus_one_one(action)
        atanh_scaled_action = torch.atanh(scaled_action)
        log_derivative_of_atanh_scaled_action = -torch_safe_log(
            1 - scaled_action**2, eps=self.safe_log_eps
        ).sum(dim=1)
        return (
            MultivariateNormal(
                loc=mean,
                covariance_matrix=self.get_parameter("cov_matrix"),
            ).log_prob(atanh_scaled_action)
            + log_derivative_of_atanh_scaled_action
        )

    def sample(self, observation):
        mean = self.forward_observation(observation)
        sampled_scaled_action = torch.tanh(
            MultivariateNormal(
                loc=mean,
                covariance_matrix=self.get_parameter("cov_matrix"),
            ).sample()
        )
        sampled_action = self.unscale_from_minus_one_one_to_action_bounds(
            sampled_scaled_action
        )

        return sampled_action


class GaussianPerceptronPDFModel(TanhGaussianPDFModel):
    def __init__(
        self,
        *args,
        hidden_size=10,
        leaky_relu_coef=0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.in_layer = nn.Linear(self.dim_observation, hidden_size, bias=False)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, self.dim_action, bias=False)
        self.leaky_relu_coef = leaky_relu_coef

    def forward_observation(self, observation):
        self.in_layer.apply(self.weight_clipper)
        self.hidden_layer.apply(self.weight_clipper)
        # self.out_layer.apply(self.weight_clipper)

        x = self.in_layer(observation)
        x = nn.LeakyReLU(self.leaky_relu_coef)(x)
        x = self.hidden_layer(x)
        x = nn.LeakyReLU(self.leaky_relu_coef)(x)
        x = self.out_layer(x)

        out = self.normalize(x)

        return out


class GaussianElementWisePDFModel(ModelNN):
    def __init__(
        self,
        dim_observation,
        diag_scale_coef,
        use_derivative=False,
        weight_min=None,
        weight_max=None,
    ):
        super().__init__()

        if use_derivative:
            dim_observation = dim_observation * 2

        self.dim_observation = dim_observation
        self.register_parameter(
            name="dot_layer",
            param=torch.nn.Parameter(
                0.1 * torch.ones(self.dim_observation),
                requires_grad=True,
            ),
        )
        self.register_parameter(
            name="cov_matrix",
            param=torch.nn.Parameter(
                torch.diag(diag_scale_coef**2 * torch.ones(dim_observation).float()),
                requires_grad=False,
            ),
        )
        self.weight_min = weight_min
        self.weight_max = weight_max

        self.cache_weights()

    def get_mean_of_action_action(self, input_tensor):
        if len(input_tensor.shape) == 1:
            observation, action = (
                input_tensor[: self.dim_observation],
                input_tensor[self.dim_observation :],
            )
        elif len(input_tensor.shape) == 2:
            observation, action = (
                input_tensor[:, : self.dim_observation],
                input_tensor[:, self.dim_observation :],
            )
        else:
            raise ValueError("Input tensor has unexpected dims")
        return self.clamp_and_multiply(observation), action

    def forward(self, input_tensor, weights=None):
        if weights is not None:
            self.update(weights)

        mean_of_action, action = self.get_mean_of_action_action(input_tensor)
        cov_matrix = [
            weight for name, weight in self.named_parameters() if name == "cov_matrix"
        ][0]
        return MultivariateNormal(
            loc=mean_of_action, covariance_matrix=cov_matrix
        ).log_prob(action)

    def clamp_and_multiply(self, observation):
        dot_layer = dict(self.named_parameters())["dot_layer"]
        dot_layer = dot_layer.clamp(min=self.weight_min, max=self.weight_max)
        if len(observation.shape) == 2:
            mean_of_action = observation * dot_layer[None, :]
        mean_of_action = observation * dot_layer

        return mean_of_action

    def sample(self, observation):
        mean_of_action = self.clamp_and_multiply(observation)
        cov_matrix = [
            weight for name, weight in self.named_parameters() if name == "cov_matrix"
        ][0]
        return MultivariateNormal(
            loc=mean_of_action, covariance_matrix=cov_matrix
        ).sample()


class ModelGaussianConditional(Model):
    """
    Gaussian probability distribution model with `weights[0]` being an expectation vector
    and `weights[1]` being a covariance matrix.
    The expectation vector can optionally be generated
    """

    model_name = "model-gaussian"

    def __init__(
        self,
        expectation_function=None,
        arg_condition=None,
        weights=None,
    ):
        self.weights = rc.array(weights)
        self.weights_init = self.weights
        self.expectation_function = expectation_function
        if arg_condition is None:
            arg_condition = []

        self.arg_condition = arg_condition
        self.arg_condition_init = arg_condition

        self.update_expectation(self.arg_condition)
        self.update_covariance()

    def update_expectation(self, arg_condition):
        self.arg_condition = arg_condition
        self.expectation = -rc.dot(arg_condition, self.weights)

    def update_covariance(self):
        self.covariance = 0.5

    def compute_gradient(self, argin):
        grad = (
            -2 * self.arg_condition * (-argin[0] + self.expectation) / self.covariance
        )
        # grad = -self.arg_condition
        return grad

    def update(self, new_weights):
        # We clip the weights here to discard the too large ones.
        # It is somewhat artificial, but convenient in practice, especially for plotting weights.
        # For clipping, we use numpy explicitly without resorting to rc
        self.weights = np.clip(new_weights, 0, 100)
        self.update_expectation(self.arg_condition_init)
        self.update_covariance()

    def sample_from_distribution(self, argin):
        self.update_expectation(argin)
        self.update_covariance()

        # As rc does not have random sampling, we use numpy here.
        return rc.array([np.random.normal(self.expectation, self.covariance)])

    def forward(self, *args, weights=None):
        return weights
