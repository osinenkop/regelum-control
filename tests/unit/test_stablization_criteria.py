from regelum.data_buffers import DataBuffer
from regelum.stopping_criterion import CoordinateMaxAbsCriterion
import numpy as np
import pytest


@pytest.fixture
def data_buffer():
    data_buffer = DataBuffer()

    for episode_id in range(3):
        values = np.vstack((np.linspace(0.0, 0.1, 20), np.linspace(0.0, 0.5, 20))).T / (
            episode_id + 1
        )
        for observation in values:
            data_buffer.push_to_end(
                observation=observation.reshape(1, -1), episode_id=episode_id
            )

    return data_buffer


def test_inf_criteria(data_buffer):
    criteria = CoordinateMaxAbsCriterion(max_abs=np.inf, n_last_observations=10)
    assert criteria(data_buffer)


def test_zero_criteria(data_buffer):
    criteria = CoordinateMaxAbsCriterion(max_abs=0.0, n_last_observations=10)
    assert not criteria(data_buffer)


def test_success_criteria(data_buffer):
    criteria = CoordinateMaxAbsCriterion(max_abs=1.0, n_last_observations=10)
    assert criteria(data_buffer)


def test_fail_criteria(data_buffer):
    criteria = CoordinateMaxAbsCriterion(max_abs=0.1, n_last_observations=10)
    assert not criteria(data_buffer)


def test_coordinate_criteria(data_buffer):
    criteria = CoordinateMaxAbsCriterion(
        max_abs=np.array([0.08, 0.4]), n_last_observations=10
    )
    assert criteria(data_buffer)


def test_episode_coordinate_criteria(data_buffer):
    criteria = CoordinateMaxAbsCriterion(
        max_abs=np.array([0.04, 0.2]), n_last_observations=10
    )
    assert not criteria(data_buffer)


def test_coordinate_criteria_fail(data_buffer):
    criteria = CoordinateMaxAbsCriterion(
        max_abs=np.array([0.08, 0.01]), n_last_observations=10
    )
    assert not criteria(data_buffer)
