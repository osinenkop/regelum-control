"""Contains models.

These can be used in system dynamics fitting, critic and other tasks.

Updates to come.

"""
# import os
# import sys

import rcognita.base

# PARENT_DIR = os.path.abspath(__file__ + "/../../")
# sys.path.insert(0, PARENT_DIR)
# CUR_DIR = os.path.abspath(__file__ + "/..")
# sys.path.insert(0, CUR_DIR)

from .__utilities import rc

import math
from abc import ABC, abstractmethod
from copy import deepcopy

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


def force_positive_def(func):
    def positive_def_wrapper(self, *args, **kwargs):
        if self.force_positive_def:
            return rc.soft_abs(func(self, *args, **kwargs))
        else:
            return func(self, *args, **kwargs)

    return positive_def_wrapper


class Model(rcognita.base.RcognitaBase, ABC):
    """Blueprint of a model."""

    def __call__(self, *args, weights=None, use_stored_weights=False):
        super().__init__()
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
        """Initialize an instance of a model."""
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
        """Assign the weights of the cached model to the active model.

        This may be needed when pytorch optimizer resulted in unsatisfactory weights, for instance.

        """
        self.update_and_cache_weights(self.cache.weights)

class ModelQuadLin(Model):
    """Quadratic-linear model."""

    model_name = "quad-lin"

    def __init__(
        self,
        dim_input,
        single_weight_min=1e-6,
        single_weight_max=1e2,
        force_positive_def=True,
    ):
        """Initialize an instance of a model with quadratic and linear terms.

        :param dim_input: input dimension
        :param single_weight_min: lower bound for every weight
        :param single_weight_max: upper bound for every weight
        :param force_positive_def: whether force positive definiteness using soft_abs function
        """
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
    """Quadratic model. May contain mixed terms."""

    model_name = "quadratic"

    def __init__(
        self,
        dim_input,
        single_weight_min=1e-6,
        single_weight_max=1e2,
        force_positive_def=True,
    ):
        """Initialize an instance of a model with quadratic terms.

        :param dim_input: input dimension
        :param single_weight_min: lower bound for every weight
        :param single_weight_max: upper bound for every weight
        :param force_positive_def: whether force positive definiteness using soft_abs function
        """
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
    """Quadratic model. May contain mixed terms."""

    model_name = "quadratic-squared"

    def forward(self, *argin, weights=None):
        result = super().forward(*argin, weights=weights)

        result = result**2 / 1e5

        return result


class ModelQuadNoMix(Model):
    """Quadratic model (no mixed terms)."""

    model_name = "quad-nomix"

    def __init__(
        self,
        dim_input,
        single_weight_min=1e-6,
        single_weight_max=1e3,
    ):
        """Initialize an instance of a model with quadratic (non-mixed) terms.

        :param dim_input: input dimension
        :param single_weight_min: lower bound for every weight
        :param single_weight_max: upper bound for every weight
        """
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
    """Quadratic model (no mixed terms)."""

    model_name = "quad-nomix"

    def __init__(
        self,
        dim_input,
        single_weight_min=1e-6,
        single_weight_max=1e2,
    ):
        """Initialize an instance of a model with quadratic (non-mixed) terms.

        :param dim_input: input dimension
        :param single_weight_min: lower bound for every weight
        :param single_weight_max: upper bound for every weight
        """
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
    """Trivial model, which is typically used in actor in which actions are being optimized directly."""

    model_name = "action-sequence"

    def __init__(self, dim_output, weights_init=None):
        """Initialize an instance of a model returns weights on call independent of input.

        :param dim_input: input dimension
        :param single_weight_min: lower bound for every weight
        :param single_weight_max: upper bound for every weight
        """
        self.dim_output = dim_output
        self.weights = weights_init
        self.weights_init = weights_init
        self.update_and_cache_weights(self.weights)

    def forward(self, *argin, weights=None):
        return weights[: self.dim_output]


class ModelQuadForm(Model):
    """Quadratic form."""

    model_name = "quad_form"

    def __init__(self, weights=None):
        """Initialize an instance of model representing quadratic form.

        :param weights: a numpy array representing matrix of quadratic form
        """
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
    """Bi-quadratic form."""

    model_name = "biquad_form"

    def __init__(self, weights):
        """Initialize an instance of biquadratic form.

        :param weights: a list of two numpy arrays representing matrices of biquadratic form
        """
        self.weights = weights

    def forward(self, *argin, weights=None):
        if len(argin) != 2:
            raise ValueError("ModelBiquadForm assumes two vector arguments!")

        vec = rc.concatenate(tuple(argin))

        result = vec.T**2 @ weights[0] @ vec**2 + vec.T @ weights[1] @ vec

        result = rc.squeeze(result)

        return result


class ModelNN(nn.Module):
    """Class of pytorch neural network models. This class is not to be used barebones.

    Instead, you should inherit from it and specify your concrete architecture.

    """

    model_name = "NN"

    @property
    def cache(self):
        """Isolate parameters of cached model from the current model."""
        return self.cached_model[0]

    def detach_weights(self):
        """Excludes the model's weights from the pytorch computation graph.

        This is needed to exclude the weights from the decision variables in optimization problems.
        An example is temporal-difference optimization, where the old critic is to be treated as a frozen model.

        """
        for variable in self.parameters():
            variable.detach_()

    def cache_weights(self, whatever=None):
        """Assign the active model weights to the cached model followed by a detach.

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
        """Transform weights as a numpy array into a dictionary compatible with pytorch."""
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
        """Assign the weights of the cached model to the active model.

        This may be needed when pytorch optimizer resulted in unsatisfactory weights, for instance.

        """
        self.update_and_cache_weights(self.cache.state_dict())

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
    """pytorch equivalent to ModelQuadNoMix."""

    def __init__(
        self,
        dim_observation,
        dim_action,
        force_positive_def=False,
    ):
        """Initialize an instance of ModelQuadNoMixTorch.

        :param dim_observation: observation dimensionality
        :param dim_action: action dimensionality
        :param force_positive_def: whether force positive definiteness using soft_abs function
        """
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
    """A neural network model for DDQN with advantage learning algorithm."""

    def __init__(
        self,
        dim_observation,
        dim_action,
        dim_hidden=40,
    ):
        """Initialize an instance of ModelDDQNAdvantage.

        :param dim_observation: observation dimensionality
        :param dim_action: action dimensionality
        :param dim_hidden: a number of neurons in hidden layers
        """
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
    """A model for value function learning."""

    def __init__(
        self,
        dim_observation,
        dim_hidden=40,
    ):
        """Initialize an instance of ModelDeepObjective.

        :param dim_observation: observation dimensionality
        :param dim_hidden: a number of neurons in hidden layers
        """
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
    """A neural network model for DDQN algorithm."""

    def __init__(
        self,
        dim_observation,
        dim_action,
        actions_grid,
        dim_hidden=40,
    ):
        """Initialize an instance of ModelDDQN.

        :param dim_observation: observation dimensionality
        :param dim_action: action dimensionality
        :param actions_grid: discretized action space
        :param dim_hidden: number of neurons in hidden layers
        """
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
    """A simple Q network used in DQN algorithms."""

    def __init__(
        self,
        dim_observation,
        dim_action,
        dim_hidden=40,
        force_positive_def=False,
        bias=False,
        leaky_relu_coef=0.2,
    ):
        """Initialize an instance of ModelDQNSimple.

        :param dim_observation: observation dimensionality
        :param dim_action: action dimensionality
        :param dim_hidden: number of neurons in hidden layers
        :param force_positive_def: whether force positive definiteness using soft_abs function
        :param bias: whether include biases into neural network or not
        :param leaky_relu_coef: coefficient of leaky ReLU
        """
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


class ModelDQN(ModelNN):
    """pytorch neural network DQN."""

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
        """Initialize a DQN model.

        :param dim_observation: dimensionality of observation
        :param dim_action: dimensionality of action
        :param dim_hidden: dimensionality of hidden layers
        :param weights: initial weights
        :param force_positive_def: whether to make forward positive definite
        :param bias: whether to use bias
        :param leaky_relu_coef: coefficient for all nn.LeakyReLU in the model
        """
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
    """Pytorch weight container for actor."""

    def __init__(self, action_init):
        """Initialize an instance of ModelWeightContainerTorch.

        :param action_init: initial action
        """
        super().__init__()

        self.p = torch.nn.Parameter(torch.tensor(action_init, requires_grad=True))

        self.double()
        self.force_positive_def = False
        self.cache_weights()

    def forward(self, observation):
        return self.weights


class LookupTable(Model):
    """A tabular model for gridworlds."""

    model_name = "lookup-table"

    def __init__(self, *dims):
        """Initialize an instance of LookupTable.

        :param dims: grid dimensionality
        """
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
    """Weight clipper for pytorch layers."""

    def __init__(self, weight_min=None, weight_max=None):
        """Initialize a weight clipper.

        :param weight_min: minimum value for weight
        :param weight_max: maximum value for weight
        """
        self.weight_min = weight_min
        self.weight_max = weight_max

    def __call__(self, module):
        # filter the variables to get the ones you want
        w = module.weight.data
        w = w.clamp(self.weight_min, self.weight_max)
        module.weight.data = w


class ModelFc(ModelNN):
    """Fully connected layer with weight clipper."""

    def __init__(
        self,
        dim_observation,
        dim_action,
        use_derivative=False,
        weight_min=None,
        weight_max=None,
    ):
        """Initialize a fully connected layer.

        :param dim_observation: dimensionality of observation
        :param dim_action: dimensionality of action
        :param use_derivative: whether the derivation of observation appended to observation
        :param weight_min: minimal weight bound
        :param weight_max: maximum weight bound
        """
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
    """Model with only one diagonal linear layer."""

    def __init__(
        self, dim_observation, weight_min=None, weight_max=None, use_derivative=False
    ):
        """Initialize a model with one diagonal linear layer.

        :param dim_observation: dimensionality of observation
        :param weight_min: minimal weight bound
        :param weight_max: maximal weight bound
        :param use_derivative: whether observation derivatives are appended to observations
        """
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


class GaussianPDFModel(ModelNN):
    r"""Model for REINFORCE Policy Gradient methods.

    Markov kernel acts like :math:`u \mid x \sim \mathcal{N}\left( f_{\theta}(x), \sigma^2 \right)`.
    :math:`f_{\theta}(x)` is the neural network with weights :math:`\theta`.
    """

    def __init__(
        self,
        dim_observation,
        dim_action,
        diag_scale_coef,
        use_derivative=False,
        weight_min=None,
        weight_max=None,
    ):
        r"""Initialize an instance of GaussianPDFModel.

        :param dim_observation: dimensionality of observation
        :param dim_action: dimensionality of action
        :param diag_scale_coef: standard deviation for distribution (parameter :math:`\sigma`)
        :param use_derivative: whether observation derivatives are appended to observations
        :param weight_min: minimal weight bound
        :param weight_max: maximal weight bound
        """
        super().__init__()

        if use_derivative:
            dim_observation = dim_observation * 2

        self.dim_observation = dim_observation
        self.in_layer = nn.Linear(self.dim_observation, dim_action, bias=False)
        self.register_parameter(
            name="cov_matrix",
            param=torch.nn.Parameter(
                torch.diag(diag_scale_coef**2 * torch.ones(dim_action).float()),
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

        mean_of_action = self.in_layer(observation)
        return mean_of_action, action

    def forward(self, input_tensor, weights=None):
        if weights is not None:
            self.update(weights)

        self.in_layer.apply(WeightClipper(self.weight_min, self.weight_max))
        mean_of_action, action = self.get_mean_of_action_action(input_tensor)
        cov_matrix = [
            weight for name, weight in self.named_parameters() if name == "cov_matrix"
        ][0]
        return MultivariateNormal(
            loc=mean_of_action, covariance_matrix=cov_matrix
        ).log_prob(action)

    def sample(self, observation):
        self.in_layer.apply(WeightClipper(self.weight_min, self.weight_max))
        mean_of_action = self.in_layer(observation)

        cov_matrix = [
            weight for name, weight in self.named_parameters() if name == "cov_matrix"
        ][0]
        return MultivariateNormal(
            loc=mean_of_action, covariance_matrix=cov_matrix
        ).sample()