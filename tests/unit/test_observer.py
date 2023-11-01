from regelum.system import (
    TwoTank,
    TwoTankWithReference,
    ThreeWheeledRobotNI,
    ThreeWheeledRobotNIWithReference,
)
from regelum.observer import ObserverReference, ObserverTrivial
import numpy as np
from regelum.__utilities import rc


def test_two_tank_composed():
    two_tank_system = TwoTank()
    two_tank_composed = TwoTankWithReference()
    reference = two_tank_composed.parameters["reference"]
    default_state = np.array([[10.0, 10.0]])
    default_action = np.array([[1.0]])

    output_2tank = two_tank_system.compute_state_dynamics(
        0, default_state, default_action, _native_dim=False
    )

    output_2tank_composed = two_tank_composed.compute_state_dynamics(
        0, default_state, default_action, _native_dim=False
    )
    assert np.allclose(output_2tank, output_2tank_composed)

    observation = two_tank_system.get_observation(0, default_state, default_action)
    observation_composed = two_tank_composed.get_observation(
        0, default_state, default_action
    )

    assert np.allclose(
        ObserverReference(reference).get_state_estimation(
            None, observation=observation_composed, action=default_action
        ),
        observation,
    )

    assert np.allclose(observation_composed, observation - reference.reshape(1, -1))


def test_3wrobot_ni_composed():
    robot = ThreeWheeledRobotNI()
    robot_with_reference = ThreeWheeledRobotNIWithReference()
    reference = robot_with_reference.parameters["reference"]
    default_state = np.array([[10.0, 10.0, 0]])
    default_action = np.array([[1.0, 1.0]])

    output_2tank = robot.compute_state_dynamics(
        0, default_state, default_action, _native_dim=False
    )

    output_2tank_composed = robot_with_reference.compute_state_dynamics(
        0, default_state, default_action, _native_dim=False
    )
    assert np.allclose(output_2tank, output_2tank_composed)

    observation = robot.get_observation(0, default_state, default_action)
    observation_composed = robot_with_reference.get_observation(
        0, default_state, default_action
    )

    assert np.allclose(
        ObserverReference(reference).get_state_estimation(
            None, observation=observation_composed, action=default_action
        ),
        observation,
    )
    assert np.allclose(observation_composed, observation - reference.reshape(1, -1))
