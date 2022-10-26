"""
This module contains model classes.
These can be used in system dynamics fitting, critic and other tasks

Updates to come.

"""
import numpy as np
import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

from utilities import rc, rej_sampling_rvs
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import OrderedDict


class ModelAbstract(ABC):
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
            return self.cache.forward(*args, weights=self.weights)

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

    def update_and_cache_weights(self, weights):
        if "cache" not in self.__dict__.keys():
            self.cache = deepcopy(self)

        self.weights = weights

        self.cache.weights = weights


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


class ModelQuadLin(ModelAbstract):
    """
    Quadratic-linear model.

    """

    model_name = "quad-lin"

    def __init__(self, input_dim, weight_min=1.0, weight_max=1e3):
        self.dim_weights = int((input_dim + 1) * input_dim / 2 + input_dim)
        self.weight_min = weight_min * np.ones(self.dim_weights)
        self.weight_max = weight_max * np.ones(self.dim_weights)
        self.weights = self.weight_min
        self.update_and_cache_weights(self.weights)

    def forward(self, *argin, weights):
        if len(vec) > 1:
            vec = rc.concatenate(tuple(argin))
        else:
            vec = argin[0]

        polynom = rc.uptria2vec(rc.outer(vec, vec))
        polynom = rc.concatenate([polynom, vec]) ** 2
        result = rc.dot(weights, polynom)

        return result


class ModelQuadratic(ModelAbstract):
    """
    Quadratic model. May contain mixed terms.

    """

    model_name = "quadratic"

    def __init__(self, input_dim, single_weight_min=1.0, single_weight_max=1e3):
        self.dim_weights = int((input_dim + 1) * input_dim / 2)
        self.weight_min = single_weight_min * np.ones(self.dim_weights)
        self.weight_max = single_weight_max * np.ones(self.dim_weights)
        self.weights = self.weight_min
        self.update_and_cache_weights(self.weights)

    def forward(self, *argin, weights):
        if len(vec) > 1:
            vec = rc.concatenate(tuple(argin))
        else:
            vec = argin[0]

        polynom = rc.to_col(rc.uptria2vec(rc.outer(vec, vec))) ** 2
        result = rc.dot(weights, polynom)

        return result


class ModelQuadNoMix(ModelAbstract):
    """
    Quadratic model (no mixed terms).

    """

    model_name = "quad-nomix"

    def __init__(self, input_dim, single_weight_min=1e-3, single_weight_max=1e3):
        self.dim_weights = input_dim
        self.weight_min = single_weight_min * np.ones(self.dim_weights)
        self.weight_max = single_weight_max * np.ones(self.dim_weights)
        self.weights = self.weight_min
        self.update_and_cache_weights(self.weights)

    def forward(self, *argin, weights=None):
        if len(argin) > 1:
            vec = rc.concatenate(tuple(argin))
        else:
            vec = argin[0]

        polynom = vec * vec
        result = rc.dot(weights, polynom)

        return result


class ModelWeightContainer(ModelAbstract):
    """
    Quadratic model (no mixed terms).

    """

    model_name = "action-sequence"

    def __init__(self, weights_init=None):
        self.weights = weights_init
        self.weights_init = weights_init
        self.update_and_cache_weights(self.weights)

    def forward(self, *argin):
        return self.weights


# class ModelQuadMix(ModelBase):
#     model_name = "quad-mix"

#     def __init__(self, input_dim, weight_min=1.0, weight_max=1e3):
#         self.dim_weights = int(
#             self.dim_output + self.dim_output * self.dim_input + self.dim_input
#         )
#         self.weight_min = weight_min * np.ones(self.dim_weights)
#         self.weight_max = weight_max * np.ones(self.dim_weights)

#     def _forward(self, vec, weights):

#         v1 = rc.to_col(v1)
#         v2 = rc.to_col(v2)

#         polynom = rc.concatenate([v1 ** 2, rc.kron(v1, v2), v2 ** 2])
#         result = rc.dot(weights, polynom)

#         return result


class ModelQuadForm(ModelAbstract):
    """
    Quadratic form.

    """

    model_name = "quad_form"

    def __init__(self, weights=None):
        self.weights = weights

    def forward(self, *argin, weights):

        if len(argin) != 2:
            raise ValueError("ModelQuadForm assumes two vector arguments!")

        vec = rc.concatenate(tuple(argin))

        result = vec.T @ weights @ vec

        result = rc.squeeze(result)

        return result


class ModelBiquadForm(ModelAbstract):
    """
    Bi-quadratic form.

    """

    model_name = "biquad_form"

    def __init__(self, weights):
        self.weights = weights

    def forward(self, *argin, weights):
        if len(argin) != 2:
            raise ValueError("ModelBiquadForm assumes two vector arguments!")
        result = (
            argin[0].T ** 2,
            weights[0] @ argin[1] ** 2 + argin[0].T @ weights[1] @ argin[1],
        )

        result = rc.squeeze(result)

        return result


class ModelNN(nn.Module):
    """
    pytorch neural network of three layers: fully connected, ReLU, fully connected.

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

    def cache_weights(self):
        """
        Assign the active model weights to the cached model followed by a detach.

        This method also backs up itself and performs this operation only once upon the initialization procedure
        """
        if "cached_model" not in self.__dict__.keys():
            self.cached_model = (
                deepcopy(self),
            )  ## this is needed to prevent cached_model's parameters to be parsed by model init hooks

        self.cache.load_state_dict(self.state_dict())
        self.cache.detach_weights()

    def update(self, weights):
        if not isinstance(weights, OrderedDict):
            weights_dict = self.weights2dict(weights)
        elif not isinstance(weights, list):
            raise TypeError("weights must be passed as either OrderedDict or list type")
        self.load_state_dict(weights_dict)

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
            weights = self.weights2dict(weights)
            self.load_state_dict(weights)
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

        argin = torch.tensor(argin)

        if use_stored_weights is False:
            if weights is not None:
                result = self.forward(argin, weights)
            else:
                result = self.forward(argin)
        else:
            result = self.cache.forward(argin)

        return result


class ModelQuadNoMixTorch(ModelNN):
    def __init__(self, dim_observation, dim_action, dim_hidden=20, weights=None):
        super().__init__()

        self.fc1 = nn.Linear(dim_observation + dim_action, 1, bias=False)

        if weights is not None:
            self.load_state_dict(weights)

        self.double()
        self.cache_weights()

    def forward(self, input_tensor, weights=None):
        if weights is not None:
            self.update(weights)

        x = input_tensor
        x = self.fc1(x)

        x = -(x ** 2)
        x = torch.sum(x)

        return x


class LookupTable(ModelAbstract):
    model_name = "lookup-table"

    def __init__(self, *dims):
        dims = tuple(
            np.concatenate(tuple([np.atleast_1d(dim) for dim in dims])).astype(int)
        )
        self.weights = rc.zeros(dims)
        self.update_and_cache_weights(self.weights)

    def __call__(self, *argin, use_stored_weights=False):

        if use_stored_weights is False:
            result = self.forward(*argin)
        else:
            result = self.cache.forward(*argin)
        return result

    def forward(self, *argin):
        indices = tuple(
            np.squeeze(
                np.concatenate(tuple([np.atleast_1d(np.array(ind)) for ind in argin]))
            ).astype(int)
        )
        return self.weights[indices]


class ModelGaussianConditional(ModelAbstract):
    """
    Gaussian probability distribution model with `weights[0]` being an expectation vector
    and `weights[1]` being a covariance matrix.
    The expectation vector can optionally be generated
    """

    model_name = "model-gaussian"

    def __init__(
        self, expectation_function=None, arg_condition=[], weights=None, jitter=1e-6,
    ):

        self.weights = np.array(weights)
        self.weights_init = self.weights
        self.expectation_function = expectation_function
        self.arg_condition = arg_condition
        self.arg_condition_init = arg_condition
        self.jitter = jitter

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
        self.weights = np.clip(new_weights, 0, 100)
        self.update_expectation(self.arg_condition_init)
        self.update_covariance()

    def sample_from_distribution(self, argin):
        self.update_expectation(argin)
        self.update_covariance()

        return np.array([np.random.normal(self.expectation, self.covariance)])

    def forward(self,):
        pass
