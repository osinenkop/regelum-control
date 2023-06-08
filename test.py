from rcognita.systems import System, ThreeWheeledRobot, TwoTank
from rcognita.planners import TwoTankPlanner
from rcognita.__utilities import rc
from rcognita.optimizers import LazyOptimizer, SciPyOptimizer, CasADiOptimizer
import numpy as np

# class ThreeWheeledRobotNI(System):
#     """
#     System class: 3-wheel robot with static actuators (the NI - non-holonomic integrator).
#     """

#     _name = "three-wheeled-robot-ni"
#     _system_type = "diff_eqn"
#     _dim_state = 3
#     _dim_inputs = 2
#     _dim_observation = 3

#     def compute_state_dynamics(self, time, state, inputs):
#         Dstate = rc.zeros(self.dim_state, prototype=(state, inputs))

#         Dstate[0] = inputs[0] * rc.cos(state[2])
#         Dstate[1] = inputs[0] * rc.sin(state[2])
#         Dstate[2] = inputs[1]

#         return Dstate


# class Integrator(System):
#     _name = "integral-parts"
#     _system_type = "diff_eqn"
#     _dim_state = 0
#     _dim_inputs = 2
#     _dim_observation = 2
#     _parameters = {"m": 10, "I": 1}

#     def compute_state_dynamics(self, time, state, inputs):
#         Dstate = rc.zeros(
#             self.dim_state,
#             prototype=(state, inputs),
#         )

#         m, I = self.parameters["m"], self.parameters["I"]

#         Dstate[0] = 1 / m * inputs[0]
#         Dstate[1] = 1 / I * inputs[1]

#         return Dstate


# system = TwoTank().compose(TwoTankPlanner(), output_mode="right")
# obs = system.get_observation(None, [0, 0.4], [1])
# print(obs)
objective = lambda x: (x - 1) ** 2 + 1
bounds = np.array([[-0.5, 0.5]])

optimizer = LazyOptimizer(
    class_object=SciPyOptimizer,
    objective_function=objective,
    decision_variable_bounds=bounds,
)


optimizer.specify_decision_variable_dimensions(decision_variable_dim=1)
optimizer.specify_opt_method("SLSQP")
optimizer.specify_parameters()
optimizer.apply_bounds(bounds)
optimizer.subject_to([lambda x: x**2 - 0.01, lambda x: x**2 + 0.01])
print(type(optimizer))
n_optimizer = optimizer.instantiate()
print(type(n_optimizer))
x = n_optimizer.optimize([0.0])
print(x)
