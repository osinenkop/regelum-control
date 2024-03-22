from regelum.system import (
    TwoTank,
    ThreeWheeledRobotKinematic,
    SystemWithConstantReference,
)
from regelum.observer import ObserverReference
import numpy as np
from regelum.callback import detach


def test_constant_reference():
    for system in (detach(TwoTank)(), detach(ThreeWheeledRobotKinematic)()):
        system_with_reference = detach(SystemWithConstantReference)(
            system, state_reference=np.arange(system.dim_state)
        )
        reference = system_with_reference.parameters["reference"]
        default_state = 10 * np.random.randn(1, system.dim_state)
        default_action = np.ones((1, system.dim_inputs))

        output_system = system_with_reference.compute_state_dynamics(
            0, default_state, default_action, _native_dim=False
        )

        output_system_with_reference = system_with_reference.compute_state_dynamics(
            0, default_state, default_action, _native_dim=False
        )
        assert np.allclose(output_system, output_system_with_reference)

        observation = system.get_observation(0, default_state, default_action)
        observation_referenced = system_with_reference.get_observation(
            0, default_state, default_action
        )

        assert np.allclose(
            ObserverReference(reference).get_state_estimation(
                None, observation=observation_referenced, action=default_action
            ),
            observation,
        )

        assert np.allclose(
            observation_referenced, observation - reference.reshape(1, -1)
        )
