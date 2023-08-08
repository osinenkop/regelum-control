"""Contains models.

These can be used in system dynamics fitting, critic and other tasks.

Updates to come.

"""
from copy import deepcopy

import rcognita


from .__utilities import rc

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

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any, Tuple
import numpy as np


def force_positive_def(func):
    def positive_def_wrapper(self, *args, **kwargs):
        if self.force_positive_def:
            return rc.soft_abs(func(self, *args, **kwargs))
        else:
            return func(self, *args, **kwargs)

    return positive_def_wrapper


class Model(rcognita.RcognitaBase, ABC):
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
    def named_parameters(self):
        return self.weights

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


class ModelQuadLinQuad(Model):
    """Quadratic-linear model."""

    model_name = "quad-lin"

    def __init__(
        self,
        dim_input,
        single_weight_min=1e-6,
        single_weight_max=1e2,
        force_positive_def=True,
    ):
        self.dim_weights = int((dim_input + 1) * dim_input / 2 + dim_input) + dim_input
        self.dim_input = dim_input
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
        result = (rc.abs(rc.dot(weights[: -self.dim_input], polynom)) + 1) * rc.sqrt(
            rc.dot(weights[-self.dim_input :], vec**2)
        )

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

    def forward(self, *argin, weights=None, device="cpu"):
        if len(argin) != 2:
            raise ValueError("ModelQuadForm assumes two vector arguments!")

        vec = rc.concatenate(tuple(argin))

        try:
            result = vec.T @ weights @ vec
        except:
            result = (
                vec.T
                @ torch.tensor(weights, requires_grad=False, device=device).float()
                @ vec
            )

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
        if len(argin) == 2:
            left, right = argin
            if len(left.shape) != len(right.shape):
                raise ValueError(
                    "In ModelNN.__call__ left and right arguments must have same number of dimensions!"
                )

            dim = len(left.shape)

            if dim == 1:
                argin = rc.concatenate(argin)
            elif dim == 2:
                argin = torch.cat(argin, dim=1)
            else:
                raise ValueError("Wrong number of dimensions in ModelNN.__call__")
        elif len(argin) == 1:
            argin = argin[0]
        else:
            raise ValueError(
                f"Wrong number of arguments in ModelNN.__call__. Can be either 1 or 2. Got: {len(argin)}"
            )

        argin = argin if isinstance(argin, torch.Tensor) else torch.tensor(argin)

        if use_stored_weights is False:
            if weights is not None:
                result = self.forward(argin, weights)
            else:
                result = self.forward(argin)
        else:
            result = self.cache.forward(argin)

        return result


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
        if self.weight_min is not None or self.weight_max is not None:
            # filter the variables to get the ones you want
            w = module.weight.data
            w = w.clamp(self.weight_min, self.weight_max)
            module.weight.data = w


# TODO: WHY IS THIS CALLED QUAD MIX BLA-BLA IF IT'S JUST ONE LAYER? FIX
class ModelQuadNoMixTorch(ModelNN):
    """pytorch equivalent to ModelQuadNoMix."""

    def __init__(
        self,
        dim_observation,
        dim_action,
        dim_hidden=20,
        weights=None,
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

    def forward(
        self, input_tensor: torch.FloatTensor, weights=None
    ) -> torch.FloatTensor:
        if weights is not None:
            self.update(weights)

        x = input_tensor**2
        x = self.w1**2 @ x

        return x


class ModelPerceptron(ModelNN):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dim_hidden: int,
        n_hidden_layers: int,
        leaky_relu_coef: float = 0.15,
        force_positive_def: bool = False,
        is_force_infinitesimal: bool = False,
        is_bias: bool = True,
        weight_max: Optional[float] = None,
        weight_min: Optional[float] = None,
        weights=None,
    ):
        ModelNN.__init__(self)
        self.weight_clipper = WeightClipper(weight_min, weight_max)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.n_hidden_layers = n_hidden_layers
        self.leaky_relu_coef = leaky_relu_coef
        self.force_positive_def = force_positive_def
        self.is_force_infinitesimal = is_force_infinitesimal
        self.is_bias = is_bias
        self.input_layer = nn.Linear(dim_input, dim_hidden, bias=is_bias)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(dim_hidden, dim_hidden, bias=is_bias)
                for _ in range(n_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(dim_hidden, dim_output, bias=is_bias)

        if weights is not None:
            self.load_state_dict(weights)

        self.cache_weights()

    def _forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        self.input_layer.apply(self.weight_clipper)
        x = nn.functional.leaky_relu(
            self.input_layer(x), negative_slope=self.leaky_relu_coef
        )
        for hidden_layer in self.hidden_layers:
            hidden_layer.apply(self.weight_clipper)
            x = nn.functional.leaky_relu(
                hidden_layer(x), negative_slope=self.leaky_relu_coef
            )
        x = self.output_layer(x)

        return x

    @force_positive_def
    def forward(
        self, input_tensor: torch.FloatTensor, weights=None
    ) -> torch.FloatTensor:
        if weights is not None:
            self.update(weights)

        if self.is_force_infinitesimal:
            return self._forward(input_tensor) - self._forward(
                torch.zeros_like(input_tensor)
            )

        return self._forward(input_tensor)


class ModelWeightContainerTorch(ModelNN):
    """Pytorch model with forward that returns weights."""

    def __init__(
        self,
        dim_weights: int,
        output_bounds: Optional[List[Any]] = None,
    ):
        """Instantiate ModelWeightContainerTorch.

        :param dim_weights: Dimensionality of the weights
        :type dim_weights: int
        :param output_bounds: Bounds of the output. If `None` output is not bounded, defaults to None
        :type output_bounds: Optional[List[Any]], optional
        """
        ModelNN.__init__(self)
        self.bounder = (
            NNOutputBounder(output_bounds) if output_bounds is not None else None
        )
        self.dim_weights = dim_weights
        self.model_weights_parameter = (
            torch.nn.Parameter(
                torch.FloatTensor(torch.zeros(self.dim_weights)),
                requires_grad=True,
            ),
        )

        self.register_parameter(
            name="model_weights",
            param=torch.nn.Parameter(
                torch.FloatTensor(torch.zeros(self.dim_weights)),
                requires_grad=True,
            ),
        )

    def forward(self, inputs):
        if len(inputs.shape) == 2:
            inputs_like = torch.tile(
                self.get_parameter("model_weights"), (inputs.shape[0], 1)
            )
            if self.bounder is not None:
                return self.bounder(inputs_like)
            else:
                return inputs_like
        elif len(inputs.shape) == 1:
            if self.bounder is not None:
                return self.bounder(self.get_parameter("model_weights"))
            else:
                return self.get_parameter("model_weights")
        else:
            raise ValueError("Wrong inputs shape! Can be either 1 or 2")


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


class NNOutputBounder(ModelNN):
    r"""Output layer for bounding the model's output. The formula is: math: `F^{-1}(\\tanh(x))`, where F is the linear transformation from `bounds` to [-1, 1]."""

    def __init__(self, bounds: Union[List[Any], np.array]):
        """Initialize an instance of NNOutputBounder.

        :param bounds: Bounds for the output.
        :type bounds: Union[List[Any], np.array]
        """
        ModelNN.__init__(self)
        self.register_parameter(
            name="bounds",
            param=torch.nn.Parameter(
                torch.FloatTensor(bounds),
                requires_grad=False,
            ),
        )

    def get_unscale_coefs_from_minus_one_one_to_bounds(
        self,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        bounds = self.get_parameter("bounds")
        unscale_bias, unscale_multiplier = (
            bounds.mean(dim=1),
            (bounds[:, 1] - bounds[:, 0]) / 2.0,
        )
        return unscale_bias, unscale_multiplier

    def unscale_from_minus_one_one_to_bounds(
        self, x: torch.FloatTensor
    ) -> torch.FloatTensor:
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_bounds()

        return x * unscale_multiplier + unscale_bias

    def scale_from_bounds_to_minus_one_one(self, y):
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_bounds()

        return (y - unscale_bias) / unscale_multiplier

    def forward(self, inputs):
        return self.unscale_from_minus_one_one_to_bounds(torch.tanh(inputs))


class PerceptronWithNormalNoise(ModelNN):
    r"""Sample from :math:`F^{-1}\\left(\\mathcal{N}(f_{\\theta}(x), \\sigma^2)\\right)`, where :math:`\\sigma` is the standard deviation of the noise, :math:`f_{\\theta}(x)` is perceptron with weights :math:`\\theta`, and :math:`F` is the linear transformation from `bounds` to [-1, 1]."""

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dim_hidden: int,
        n_hidden_layers: int,
        leaky_relu_coef: float,
        output_bounds: Union[List[Any], np.array],
        sigma: float,
        normalize_output_coef: float,
        weight_min: Optional[float] = None,
        weight_max: Optional[float] = None,
    ):
        r"""Instantiate PerceptronWithNormalNoise.

        :param dim_input: Dimensionality of input (x)
        :type dim_input: int
        :param dim_output: Dimensionality of output :math `f_{\\theta}(x)`
        :type dim_output: int
        :param dim_hidden: Dimensionality of hidden layers in perceptron :math `f_{\\theta}(x)`
        :type dim_hidden: int
        :param n_hidden_layers: Number of hidden layers in perceptron :math `f_{\\theta}(x)`
        :type n_hidden_layers: int
        :param leaky_relu_coef: Negative slope of the nn.LeakyReLU in perceptron.
        :type leaky_relu_coef: float
        :param output_bounds: Bounds for the output
        :type output_bounds: Union[List[Any], np.array]
        :param sigma: Standard deviation of normal distribution
        :type sigma: float
        :param normalize_output_coef: Coefficient :math `L` in latest activation function in perceptron :math `(1 - 3 \\sigma)\\tanh\\left(\\frac{\\cdot}{L}\\right)`. We use :math `3\\sigma` rule here to guarantee that sampled random variable is in [-1, 1] with good probability. Moreover, :math `L` is an hyperparameter that stabilizes the training in small times.
        :type normalize_output_coef: float
        :param weight_min: Minimum value for weight. If `None` the weights are not clipped, defaults to None
        :type weight_min: Optional[float], optional
        :param weight_max: Maximum value for weight. If `None` the weights are not clipped, defaults to None
        :type weight_max: Optional[float], optional
        """
        super().__init__()
        self.std = sigma
        self.normalize_output_coef = normalize_output_coef

        self.perceptron = ModelPerceptron(
            dim_input=dim_input,
            dim_output=dim_output,
            dim_hidden=dim_hidden,
            n_hidden_layers=n_hidden_layers,
            leaky_relu_coef=leaky_relu_coef,
            weight_min=weight_min,
            weight_max=weight_max,
        )

        self.bounder = NNOutputBounder(output_bounds)

        self.register_parameter(
            name="scale_tril_matrix",
            param=torch.nn.Parameter(
                (self.std * torch.eye(dim_output)).float(),
                requires_grad=False,
            ),
        )
        self.cache_weights()

    def get_mean(self, observations):
        assert 1 - 3 * self.std > 0, "1 - 3 std should be greater than 0"
        # We should guarantee with good probability that sampled actions are within action bounds that are scaled to [-1, 1]
        # That is why we use 3 sigma rule here
        return (1 - 3 * self.std) * torch.tanh(
            self.perceptron(observations) / self.normalize_output_coef
        )

    def forward(self, observations):
        return self.bounder.unscale_from_minus_one_one_to_bounds(
            self.get_mean(observations)
        )

    def log_pdf(self, observations, actions):
        means = self.get_mean(observations)
        scaled_actions = self.bounder.scale_from_bounds_to_minus_one_one(actions)

        return MultivariateNormal(
            loc=means,
            scale_tril=self.get_parameter("scale_tril_matrix"),
        ).log_prob(scaled_actions)

    def sample(self, observation):
        mean = self.get_mean(observation)
        sampled_scaled_action = MultivariateNormal(
            loc=mean,
            scale_tril=self.get_parameter("scale_tril_matrix"),
        ).sample()

        sampled_action = self.bounder.unscale_from_minus_one_one_to_bounds(
            sampled_scaled_action
        )

        return sampled_action
