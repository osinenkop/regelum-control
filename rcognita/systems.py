"""
This module contains a generic interface for systems (environments) as well as concrete systems as realizations of the former

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np

import rcognita
import rcognita.base
from abc import ABC, abstractmethod
from .__utilities import rc


class System(rcognita.base.RcognitaBase, ABC):
    """
     Interface class of dynamical systems a.k.a. environments.
     Concrete systems should be built upon this class.
     To design a concrete system: inherit this class, override:
         | :func:`~systems.system.compute_state_dynamics` :
         | right-hand side of system description (required)
         | :func:`~systems.system._dynamic_control` :
         | right-hand side of controller dynamical model (if necessary)
         | :func:`~systems.system.get_observation` :
         | system out (if not overridden, output is identical to state)

     Attributes
     ----------
     system_type : : string
         Type of system by description:

         | ``diff_eqn`` : differential equation :math:`\mathcal D state = f(state, action, disturb)`
         | ``discr_fnc`` : difference equation :math:`state^+ = f(state, action, disturb)`
         | ``discr_prob`` :  by probability distribution :math:`X^+ \sim P_X(state^+| state, action, disturb)`

     where:

         | :math:`state` : state
         | :math:`action` : input
         | :math:`disturb` : disturbance

     The time variable ``time`` is commonly used by ODE solvers, and you shouldn't have it explicitly referenced in the definition, unless your system is non-autonomous.
     For the latter case, however, you already have the input and disturbance at your disposal.

     Parameters of the system are contained in ``pars`` attribute.

     dim_state, dim_input, dim_output, dim_disturb : : integer
         System dimensions
     pars : : list
         List of fixed parameters of the system
     action_bounds : : array of shape ``[dim_input, 2]``
         Box control constraints.
         First element in each row is the lower bound, the second - the upper bound.
         If empty, control is unconstrained (default)
     is_dynamic_controller : : 0 or 1
         If 1, the controller (a.k.a. agent) is considered as a part of the full state vector
     is_disturb : : 0 or 1
         If 0, no disturbance is fed into the system
     pars_disturb : : list
         Parameters of the disturbance model

    Each concrete system must realize ``System`` and define ``name`` attribute.

    """

    _name = None
    _system_type = None
    _dim_state = None
    _dim_action = None
    _dim_observation = None
    _system_parameters = {}

    def __init__(
        self,
        system_parameters_init={},
        state_init=None,
        action_init=None,
    ):
        """
        Parameters
        ----------
        system_type : : string
            Type of system by description:

            | ``diff_eqn`` : differential equation :math:`\mathcal D state = f(state, action, disturb)`
            | ``discr_fnc`` : difference equation :math:`state^+ = f(state, action, disturb)`
            | ``discr_prob`` :  by probability distribution :math:`X^+ \sim P_X(state^+| state, action, disturb)`

        where:

            | :math:`state` : state
            | :math:`action` : input
            | :math:`disturb` : disturbance

        The time variable ``time`` is commonly used by ODE solvers, and you shouldn't have it explicitly referenced in the definition, unless your system is non-autonomous.
        For the latter case, however, you already have the input and disturbance at your disposal.

        Parameters of the system are contained in ``pars`` attribute.

        dim_state, dim_input, dim_output, dim_disturb : : integer
            System dimensions
        pars : : list
            List of fixed parameters of the system
        action_bounds : : array of shape ``[dim_input, 2]``
            Box control constraints.
            First element in each row is the lower bound, the second - the upper bound.
            If empty, control is unconstrained (default)
        is_dynamic_controller : : 0 or 1
            If 1, the controller (a.k.a. agent) is considered as a part of the full state vector
        is_disturb : : 0 or 1
            If 0, no disturbance is fed into the system
        pars_disturb : : list
            Parameters of the disturbance model
        """

        assert self.system_type, "class.system_type should be set"
        assert self.dim_state, "class.dim_state should be set"
        assert self.dim_action, "class.dim_action should be set"
        assert self.dim_observation, "class.dim_observation should be set"
        assert isinstance(system_parameters_init, dict)

        if system_parameters_init:
            self._system_parameters.update(system_parameters_init)

        self.system_parameters_init = self._system_parameters

        if state_init is None:
            self.state = rc.zeros(self.dim_state)
        else:
            self.state = state_init

        if action_init is None:
            self.action = rc.zeros(self.dim_action)
        else:
            self.action = action_init

    @property
    def name(self):
        return self._name

    @property
    def system_type(self):
        return self._system_type

    @property
    def dim_state(self):
        return self._dim_state

    @property
    def dim_observation(self):
        return self._dim_observation

    @property
    def dim_action(self):
        return self._dim_action

    @property
    def system_parameters(self):
        return self._system_parameters

    @abstractmethod
    def compute_state_dynamics(self, time, state, action):
        """
        Description of the system internal dynamics.
        Depending on the system type, may be either the right-hand side of the respective differential or difference equation, or a probability distribution.
        As a probability disitribution, ``compute_state_dynamics`` should return a number in :math:`[0,1]`

        """
        pass

    def get_observation(self, time, state, action):
        """
        System output.
        This is commonly associated with signals that are measured in the system.
        Normally, output depends only on state ``state`` since no physical processes transmit input to output instantly.

        See also
        --------
        :func:`~systems.system.compute_state_dynamics`

        """
        # Trivial case: output identical to state

        return state

    def receive_action(self, action):
        """
        Receive exogeneous control action to be fed into the system.
        This action is commonly computed by your controller (agent) using the system output :func:`~systems.system.get_observation`.

        Parameters
        ----------
        action : : array of shape ``[dim_input, ]``
            Action

        """
        self.action = action

    def receive_state(self, state):
        self.state = state

    def update_system_parameters(self, inputs):
        assert isinstance(inputs, dict)
        self._system_parameters.update(inputs)
        return self.system_parameters

    def compute_closed_loop_rhs(self, time, state):
        """
        Right-hand side of the closed-loop system description.
        Combines everything into a single vector that corresponds to the right-hand side of the closed-loop system description for further use by simulators.

        Attributes
        ----------
        state_full : : vector
            Current closed-loop system state

        """
        action = self.action

        rhs_full_state = self.compute_state_dynamics(time, state, action)

        return rhs_full_state

    def reset(self):
        self.update_system_parameters(self.system_parameters_init)


class KinematicPoint(System):
    _name = "kinematic-point"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_action = 2
    _dim_observation = 2

    def compute_state_dynamics(self, time, state, action):
        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, action),
        )

        for i in range(rc.shape(action)[0]):
            Dstate[i] = action[i]

        return Dstate


class InvertedPendulumPID(System):
    """
    System class: mathematical pendulum

    """

    _name = "inverted-pendulum"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_action = 2
    _dim_observation = 3
    _system_parameters = {"m": 1, "g": 9.8, "l": 1}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.time_old = 0
        self.integral_alpha = 0

    def compute_state_dynamics(self, time, state, action):
        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, action),
        )

        m, g, l = (
            self.system_parameters["m"],
            self.system_parameters["g"],
            self.system_parameters["l"],
        )

        Dstate[0] = state[1]
        Dstate[1] = g / l * rc.sin(state[0]) + action[0] / (m * l**2)

        return Dstate

    def get_observation(self, time, state, action):
        delta_time = time - self.time_old if time is not None else 0
        self.integral_alpha += delta_time * state[0]

        return rc.array([state[0], self.integral_alpha, state[1]])

    def reset(self):
        self.time_old = 0
        self.integral_alpha = 0


class InvertedPendulumPD(InvertedPendulumPID):
    _dim_observation = 2

    def get_observation(self, time, state, action):
        return rc.array([state[0], state[1]])


class ThreeWheeledRobot(System):
    """
    System class: 3-wheeled robot with dynamical actuators.

    Description
    -----------
    Three-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_]

    .. math::
        \\begin{array}{ll}
                        \dot x_с & = v \cos \\angle \\newline
                        \dot y_с & = v \sin \\angle \\newline
                        \dot \\angle & = \\omega \\newline
                        \dot v & = \\left( \\frac 1 m F + q_1 \\right) \\newline
                        \dot \\omega & = \\left( \\frac 1 I M + q_2 \\right)
        \\end{array}

    **Variables**

    | :math:`x_с` : state-coordinate [m]
    | :math:`y_с` : observation-coordinate [m]
    | :math:`\\angle` : turning angle [rad]
    | :math:`v` : speed [m/s]
    | :math:`\\omega` : revolution speed [rad/s]
    | :math:`F` : pushing force [N]
    | :math:`M` : steering torque [Nm]
    | :math:`m` : robot mass [kg]
    | :math:`I` : robot moment of inertia around vertical axis [kg m\ :sup:`2`]
    | :math:`disturb` : actuator disturbance (see :func:`~RLframe.system.disturbDyn`). Is zero if ``is_disturb = 0``

    :math:`state = [x_c, y_c, \\angle, v, \\omega]`

    :math:`action = [F, M]`

    ``pars`` = :math:`[m, I]`

    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
        nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    """

    _name = "three-wheeled-robot"
    _system_type = "diff_eqn"
    _dim_state = 5
    _dim_action = 2
    _dim_observation = 5
    _system_parameters = {"m": 10, "I": 1}

    def compute_state_dynamics(self, time, state, action):
        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, action),
        )

        m, I = self.system_parameters["m"], self.system_parameters["I"]

        Dstate[0] = state[3] * rc.cos(state[2])
        Dstate[1] = state[3] * rc.sin(state[2])
        Dstate[2] = state[4]
        Dstate[3] = 1 / m * action[0]
        Dstate[4] = 1 / I * action[1]

        return Dstate


class ThreeWheeledRobotNI(System):
    """
    System class: 3-wheel robot with static actuators (the NI - non-holonomic integrator).
    """

    _name = "three-wheeled-robot-ni"
    _system_type = "diff_eqn"
    _dim_state = 3
    _dim_action = 2
    _dim_observation = 3

    def compute_state_dynamics(self, time, state, action):
        Dstate = rc.zeros(self.dim_state, prototype=(state, action))

        Dstate[0] = action[0] * rc.cos(state[2])
        Dstate[1] = action[0] * rc.sin(state[2])
        Dstate[2] = action[1]

        return Dstate


class TwoTank(System):
    """
    Two-tank system with nonlinearity.

    """

    _name = "two-tank"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_action = 1
    _dim_observation = 3
    _system_parameters = {"tau1": 18.4, "tau2": 24.4, "K1": 1.3, "K2": 1.0, "K3": 0.2}

    def compute_state_dynamics(self, time, state, action):
        tau1, tau2, K1, K2, K3 = (
            self.system_parameters["tau1"],
            self.system_parameters["tau2"],
            self.system_parameters["K1"],
            self.system_parameters["K2"],
            self.system_parameters["K3"],
        )

        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, action),
        )
        Dstate[0] = 1 / (tau1) * (-state[0] + K1 * action[0])
        Dstate[1] = 1 / (tau2) * (-state[1] + K2 * state[0] + K3 * state[1] ** 2)

        return Dstate


class GridWorld(System):
    """
    A simple 2-dimensional grid world with five actions: left, right, up, down and do nothing.
    The action encoding rule is as follows: right, left, up, down, do nothing -> 0, 1, 2, 3, 4.

    """

    def __init__(self, dims, terminal_state):
        self.dims = dims
        self.terminal_state = terminal_state

    def compute_state_dynamics(self, current_state, action):
        if tuple(self.terminal_state) == tuple(current_state):
            return current_state
        if action == 0:
            if current_state[1] < self.dims[1] - 1:
                return (current_state[0], current_state[1] + 1)
        elif action == 2:
            if current_state[0] > 0:
                return (current_state[0] - 1, current_state[1])
        elif action == 1:
            if current_state[1] > 0:
                return (current_state[0], current_state[1] - 1)
        elif action == 3:
            if current_state[0] < self.dims[0] - 1:
                return (current_state[0] + 1, current_state[1])
        return current_state


class CartPole(System):
    """
    Cart pole system without friction. link:
    https://coneural.org/florian/papers/05_cart_pole.pdf

    """

    _name = "cartpole"
    _system_type = "diff_eqn"
    _dim_state = 4
    _dim_action = 1
    _dim_observation = 4
    _system_parameters = {"m_c": 0.1, "m_p": 2.0, "g": 9.81, "l": 0.5}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_state_dynamics(self, time, state, action, disturb=None):
        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, action),
        )

        m_c, m_p, g, l = (
            self.system_parameters["m_c"],
            self.system_parameters["m_p"],
            self.system_parameters["g"],
            self.system_parameters["l"],
        )
        theta = state[0]
        theta_dot = state[2]
        x_dot = state[3]

        sin_theta = rc.sin(theta)
        cos_theta = rc.cos(theta)

        # Dstate[0] = theta_dot

        # Dstate[1] = x_dot

        # Dstate[2] = (
        #     (
        #         g * rc.sin(theta)
        #         - rc.cos(theta)
        #         * (action[0] + m_p * l * theta_dot**2 * rc.sin(theta))
        #         / (m_c + m_p)
        #     )
        #     / l
        #     / (4 / 3 - m_p * (rc.cos(theta) ** 2) / (m_c + m_p))
        # )
        # Dstate[3] = (
        #     action[0]
        #     + m_p
        #     * l
        #     * (
        #         theta_dot**2 * rc.sin(theta)
        #         - Dstate[0] * rc.cos(theta)
        #     )
        # ) / (m_c + m_p)

        Dstate[0] = theta_dot

        Dstate[1] = x_dot

        Dstate[3] = (
            -m_p * g * cos_theta * sin_theta
            - m_p * l * theta_dot**2 * sin_theta
            + action[0]
        ) / (m_c + m_p * sin_theta**2)

        Dstate[2] = -g / l * sin_theta + Dstate[3] / l * cos_theta

        return Dstate

    def get_observation(self, time, state, action):
        theta = state[0]
        x = state[1]
        theta_dot = state[2]
        x_dot = state[3]

        theta_observed = theta - rc.floor(theta / (2 * np.pi)) * 2 * np.pi
        if theta_observed > np.pi:
            theta_observed = theta_observed - 2 * np.pi

        return rc.array([theta_observed, x, theta_dot, x_dot])


class LunarLander(System):
    """
    Lunar lander system. link:
    https://web.aeromech.usyd.edu.au/AMME3500/Course_documents/material/tutorials/Assignment%204%20Lunar%20Lander%20Solution.pdf

    """

    _name = "lander"
    _system_type = "diff_eqn"
    _dim_state = 6
    _dim_action = 2
    _dim_observation = 6
    _system_parameters = {"m": 10, "J": 3.0, "g": 1.625, "a": 1, "r": 0.5}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_landed = False

    def compute_state_dynamics(self, time, state, action, disturb=None):
        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, action),
        )

        m, J, g = (
            self.system_parameters["m"],
            self.system_parameters["J"],
            self.system_parameters["g"],
        )

        theta = state[2]
        x_dot = state[3]
        y_dot = state[4]
        theta_dot = state[5]

        left_support, right_support = self.compute_supports_geometry(state[:2], theta)

        F_l = action[0]
        F_t = action[1]

        Dstate[0] = x_dot

        Dstate[1] = y_dot

        Dstate[2] = theta_dot

        Dstate[3] = 1 / m * (F_l * rc.cos(theta) - F_t * rc.sin(theta))

        Dstate[4] = 1 / m * (F_l * rc.sin(theta) + F_t * rc.cos(theta)) - g

        Dstate[5] = (4 * F_l) / J

        # Check if any of the two lander's supports touched the ground. If yes, freeze the state.
        self.is_landed = rc.if_else(left_support[1] <= 0, 1, 0) * rc.if_else(
            right_support[1] <= 0, 1, 0
        )

        Dstate = Dstate * (1 - self.is_landed)

        return Dstate

    def compute_supports_geometry(self, xi, theta):
        A = rc.zeros((2, 2), prototype=xi)
        xi_2 = rc.zeros(2, prototype=xi)
        xi_3 = rc.zeros(2, prototype=xi)

        A[0, 0] = rc.cos(theta)
        A[0, 1] = -rc.sin(theta)
        A[1, 0] = rc.sin(theta)
        A[1, 1] = rc.cos(theta)

        a, r = self.system_parameters["a"], self.system_parameters["r"]
        xi_2[0] = xi[0] - a
        xi_2[1] = xi[1] - r
        xi_3[0] = xi[0] + a
        xi_3[1] = xi[1] - r

        xi_2_d = xi_2 - xi
        xi_3_d = xi_3 - xi

        xi_2_d_rot = A @ xi_2_d
        xi_3_d_rot = A @ xi_3_d
        xi_2_new = xi + xi_2_d_rot
        xi_3_new = xi + xi_3_d_rot
        return xi_2_new, xi_3_new

    def compute_reaction(self, r, r_support):
        m, J, g = self.pars
        lvl = r_support[1]
        e = (r - r_support) / rc.sqrt(rc.norm_2(r - r_support))
        reaction = rc.if_else(
            lvl <= 0,
            e * rc.dot(e, m * g * rc.array([0, 1])) * lvl * self.sigma,
            rc.array([0.0, 0.0]),
        )
        return -reaction
