from regelum.system import (
    ThreeWheeledRobotNI,
    ThreeWheeledRobot,
    Integrator,
    InvertedPendulum,
)
from regelum.utilis import rg
import numpy as np

robot = ThreeWheeledRobotNI()
robot_big = ThreeWheeledRobot()
robot_composed = Integrator() @ robot
robot_composed.permute_state([3, 4, 0, 1, 2])
pendulum = InvertedPendulum()


def get_sample_model_dstate(system):
    dim_state = system.dim_state
    dim_inputs = system.dim_inputs

    state_num = rg.array([i for i in range(dim_state)])
    action_num = rg.array([i for i in range(dim_inputs)])

    state_symb = rg.array_symb((dim_state, 1), literal="s")
    action_symb = rg.array_symb((dim_inputs, 1), literal="a")
    dstate_num = system.compute_state_dynamics(
        time=None, state=state_num, inputs=action_num
    )
    dstate_num_native = system.compute_state_dynamics(
        time=None, state=state_num, inputs=action_num, _native_dim=True
    )
    dstate_symb = system.compute_state_dynamics(
        time=None, state=state_symb, inputs=action_symb
    )
    dstate_symb_native = system.compute_state_dynamics(
        time=None, state=state_symb, inputs=action_symb, _native_dim=True
    )
    return (
        dstate_num,
        dstate_num_native,
        dstate_symb,
        dstate_symb_native,
        state_num,
        action_num,
        state_symb,
        action_symb,
    )


def get_sample_model_observation(system):
    dim_state = system.dim_state
    dim_inputs = system.dim_inputs

    state_num = rg.array([i for i in range(dim_state)])
    action_num = rg.array([i for i in range(dim_inputs)])

    state_symb = rg.array_symb((dim_state, 1), literal="s")
    action_symb = rg.array_symb((dim_inputs, 1), literal="a")
    observation_num = system.get_observation(
        time=None, state=state_num, inputs=action_num
    )
    observation_num_native = system.get_observation(
        time=None, state=state_num, inputs=action_num, _native_dim=True
    )
    observation_symb = system.get_observation(
        time=None, state=state_symb, inputs=action_symb
    )
    observation_symb_native = system.get_observation(
        time=None, state=state_symb, inputs=action_symb, _native_dim=True
    )
    return (
        observation_num,
        observation_num_native,
        observation_symb,
        observation_symb_native,
        state_num,
        action_num,
        state_symb,
        action_symb,
    )


def simple_unit_test(system):
    (
        dstate_num,
        dstate_num_native,
        dstate_symb,
        dstate_symb_native,
        state_num,
        action_num,
        state_symb,
        action_symb,
    ) = get_sample_model_dstate(system)

    func = rg.to_casadi_function(dstate_symb, state_symb, action_symb)

    func_native = rg.to_casadi_function(dstate_symb_native, state_symb, action_symb)
    return all(
        [
            np.allclose(func(state_num, action_num).full(), dstate_num),
            np.allclose(
                func_native(state_num, action_num).full().T.squeeze(), dstate_num_native
            ),
        ]
    )


assert all(
    [
        simple_unit_test(robot_composed),
        simple_unit_test(robot_big),
        simple_unit_test(robot),
        simple_unit_test(pendulum),
    ]
)
