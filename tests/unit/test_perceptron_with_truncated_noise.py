from regelum.model import (
    PerceptronWithTruncatedNormalNoise,
    BoundsHandler,
    MultiplyByConstant,
)
import torch
import pytest
import numpy as np
from pytest_lazyfixture import lazy_fixture


@pytest.fixture
def truncated_model():
    return PerceptronWithTruncatedNormalNoise(
        dim_input=3,
        dim_output=2,
        dim_hidden=2,
        n_hidden_layers=1,
        output_bounds=[[-15, 20], [-200, 300]],
        stds=[0.01, 1.0],
        is_truncated_to_output_bounds=True,
        linear_weights_init=torch.nn.init.constant_,
        linear_weights_init_kwargs={"val": 100.0},
        biases_init=torch.nn.init.constant_,
        biases_init_kwargs={"val": 0.0},
    )


@pytest.fixture
def untruncated_model():
    return PerceptronWithTruncatedNormalNoise(
        dim_input=3,
        dim_output=2,
        dim_hidden=2,
        n_hidden_layers=1,
        output_activation=torch.nn.Sequential(
            torch.nn.Tanh(),
            MultiplyByConstant(0.7),
            BoundsHandler([[-5, 5], [-25, 25]]),
        ),
        stds=[5 / 40.0, 25 / 40.0],
        is_truncated_to_output_bounds=False,
        linear_weights_init=torch.nn.init.constant_,
        linear_weights_init_kwargs={"val": 100.0},
        biases_init=torch.nn.init.constant_,
        biases_init_kwargs={"val": 0.0},
    )


@pytest.fixture
def right_bound_inputs():
    big_positive_inputs = 100.0 * torch.ones(1, 3)
    return big_positive_inputs


@pytest.fixture
def left_bound_inputs():
    big_negative_inputs = -100.0 * torch.ones(1, 3)
    return big_negative_inputs


@pytest.fixture
def middle_inputs():
    return torch.zeros(1, 3)


@pytest.fixture
def right_bound_tr_samples(truncated_model, right_bound_inputs):
    return torch.vstack([truncated_model(right_bound_inputs) for _ in range(1500)])


@pytest.fixture
def left_bound_tr_samples(truncated_model, left_bound_inputs):
    return torch.vstack([truncated_model(left_bound_inputs) for _ in range(1500)])


@pytest.fixture
def middle_tr_samples(truncated_model, middle_inputs):
    return torch.vstack([truncated_model(middle_inputs) for _ in range(1500)])


@pytest.fixture
def middle_un_samples(untruncated_model, middle_inputs):
    return torch.vstack([untruncated_model(middle_inputs) for _ in range(1500)])


@pytest.fixture
def right_un_samples(untruncated_model, right_bound_inputs):
    return torch.vstack([untruncated_model(right_bound_inputs) for _ in range(1500)])


@pytest.fixture
def right_bounds(truncated_model):
    return torch.FloatTensor(truncated_model.output_bounds_array[:, 1])


@pytest.fixture
def left_bounds(truncated_model):
    return torch.FloatTensor(truncated_model.output_bounds_array[:, 0])


@pytest.mark.parametrize(
    "samples",
    [
        lazy_fixture("right_bound_tr_samples"),
        lazy_fixture("left_bound_tr_samples"),
        lazy_fixture("middle_tr_samples"),
    ],
)
def test_is_within_bounds(truncated_model, samples):
    bounds = truncated_model.output_bounds_array
    assert (
        (samples[:, 0] >= bounds[0, 0]).all()
        and (samples[:, 0] <= bounds[0, 1]).all()
        and (samples[:, 1] >= bounds[1, 0]).all()
        and (samples[:, 1] <= bounds[1, 1]).all()
    )


@pytest.mark.parametrize(
    "samples, closest_bounds",
    [
        (lazy_fixture("right_bound_tr_samples"), lazy_fixture("right_bounds")),
        (lazy_fixture("left_bound_tr_samples"), lazy_fixture("left_bounds")),
    ],
)
def test_stats_on_bounds(truncated_model, samples, closest_bounds):
    three_sigma_probs = (
        (torch.abs(samples - closest_bounds) <= 3 * truncated_model.stds)
        .float()
        .mean(axis=0)
    )

    assert torch.allclose(
        three_sigma_probs, torch.full_like(three_sigma_probs, 0.997), rtol=0.05
    ), "Three sigma rule violated"

    one_sigma_probs = (
        (torch.abs(samples - closest_bounds) <= truncated_model.stds)
        .float()
        .mean(axis=0)
    )

    assert torch.allclose(
        one_sigma_probs, torch.full_like(one_sigma_probs, 0.6827), rtol=0.05
    ), "One sigma rule violated"


@pytest.mark.parametrize(
    "model, inputs, samples",
    [
        (
            lazy_fixture("truncated_model"),
            lazy_fixture("middle_inputs"),
            lazy_fixture("middle_tr_samples"),
        ),
        (
            lazy_fixture("untruncated_model"),
            lazy_fixture("right_bound_inputs"),
            lazy_fixture("right_un_samples"),
        ),
    ],
)
def test_stats_inside_bounds(model, inputs, samples):
    stds = samples.std(axis=0)
    means = samples.mean(axis=0)
    assert torch.allclose(stds, model.stds, rtol=0.05)
    assert torch.allclose(model(inputs), means, rtol=0.05)


@pytest.mark.parametrize(
    "model, samples, inputs",
    [
        (
            lazy_fixture("truncated_model"),
            lazy_fixture("middle_tr_samples"),
            lazy_fixture("middle_inputs"),
        ),
        (
            lazy_fixture("truncated_model"),
            lazy_fixture("right_bound_tr_samples"),
            lazy_fixture("right_bound_inputs"),
        ),
        (
            lazy_fixture("truncated_model"),
            lazy_fixture("left_bound_tr_samples"),
            lazy_fixture("left_bound_inputs"),
        ),
        (
            lazy_fixture("untruncated_model"),
            lazy_fixture("middle_un_samples"),
            lazy_fixture("middle_inputs"),
        ),
        (
            lazy_fixture("untruncated_model"),
            lazy_fixture("right_un_samples"),
            lazy_fixture("right_bound_inputs"),
        ),
    ],
)
def test_log_probs(model, samples, inputs):
    probs_estimates, xedges, yedges = np.histogram2d(
        samples.detach().numpy()[:, 0],
        samples.detach().numpy()[:, 1],
        bins=10,
        density=True,
    )
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    mesh_x, mesh_y = np.meshgrid(xcenters, ycenters)

    right_arg = torch.FloatTensor(np.vstack((mesh_x.reshape(-1), mesh_y.reshape(-1))).T)
    left_arg = torch.ones(right_arg.shape[0], 3) * inputs
    probs = torch.exp(model.log_pdf(left_arg, right_arg)).detach().numpy()

    assert np.allclose(probs_estimates.sum(), probs.sum(), rtol=0.05)
