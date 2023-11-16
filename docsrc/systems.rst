*****************
Avaliable systems
*****************

Inverted pendulum
=================

Description
-----------

The system consists of a pendulum attached at one end to a fixed point, and the other end being free. The pendulum is initially
pointed downwards and the goal is to apply torque :math:`M` to the joint to swing the pendulum into an upright position, with its
center of gravity right above the fixed point. Below :math:`l`, :math:`m`, :math:`g`` respectively denote the length of the pendulum, the mass of the
pendulum and the acceleration of gravity. 

.. math::

    \begin{aligned}
        & \dot{\vartheta} = \omega, \\
        & \dot{\omega} = \frac{g}{l} \sin(\vartheta) + \frac{M}{m l_p^2}.
    \end{aligned}

**Variables**

- :math:`\vartheta` : pendulum angle [rad]
- :math:`\omega` : angular velocity [rad/s]
- :math:`M` : torque [N :math:`\times` m]

**State:** :math:`\vartheta, \omega`

**Action:**: :math:`M`

**Action bounds:** :math:`M \in [-20, 20]`


Default configuration of simulator
----------------------------------

- The duration of one episode is set to 10 seconds
- The sampling rate of the pipeline is set to 100 Hz
- Initial state is set to 

.. math::

    \begin{aligned}
        & \vartheta(0) = \pi, \\
        & \omega(0) = 0
    \end{aligned}

Default running cost
--------------------

.. math::
    
    c(\vartheta, \omega, M) = 10 \vartheta ^ 2 + 3 \omega ^ 2


Usage 
-----

.. code-block:: python

    import numpy as np    
    from regelum.model import ModelQuadLin
    from regelum.system import InvertedPendulumPD
    from regelum.objective import RunningObjective
    from regelum.simulator import CasADi

    sampling_time = 0.01
    simulator = CasADi(
        system=InvertedPendulumPD(),
        state_init=np.array([[np.pi, 0.]]),
        time_final=10,
        max_step=sampling_time / 10.,
    )
    running_cost = RunningObjective(
        model=ModelQuadLin("diagonal", weights=[10., 3., 0.])
    )

Two-tank system
===============

Description
-----------

The two-tank systems consists of two tanks which are connected by an open valve. Liquid inflow :math:`U` to the first tank is
governed by a pump and there is an interconnection between the tanks. In addition, there is a permanent leak from tank 2. The
goal is to keep both tanks exactly 40% full. Below :math:`\tau_1, \tau_2, Q_1, Q_2`  respectively denote base area of the first tank, base area
of the second tank and two coefficients that determine the magnitude of inbound and outbound flow for the second tank. The
duration of one episode is set to 80 seconds, while the sampling rate of the pipeline is set to 10 Hz.

.. math::

    \begin{aligned}
        & \dot{h_1} = \frac{1}{\tau_1}(U - Q_1 h1), \\
        & \dot{h_2} = \frac{1}{\tau_2} (Q_1 h1 - Q_2 h_2 + Q_2 h_2 ^ 2).
    \end{aligned}

**Variables**

- :math:`h_1` : height in the first tank 
- :math:`h_2` : relative height in the second tank 
- :math:`\tau_1` : base area of the first tank [m * m]
- :math:`\tau_2` : base area of the second tank [m * m]
- :math:`Q_1` : outbound flow in the first tank [m^3 / s]
- :math:`Q_2` : inbound flow in the second tank [m^3 / s]
- :math:`U` : inflow to the first tank [m^3 / s]

**State:** :math:`h_1, h_2`

**Action:**: :math:`U`

**Action bounds:** :math:`U \in [0., 1]`


Default configuration of simulator
----------------------------------

- The duration of one episode is set to 80 seconds
- The sampling rate of the pipeline is set to 10 Hz
- Initial state is set to 

.. math::

    \begin{aligned}
        & h_1(0) = 2, \\
        & h_2(0) = -2
    \end{aligned}

Default running cost
--------------------

.. math::
    
    c(\h_1, \h_2, U) = 10 (\h_1 - 0.4) ^ 2 + 10 (\h_2 - 0.4) ^ 2


Usage 
-----

.. code-block:: python

    import numpy as np    
    from regelum.model import ModelQuadLin
    from regelum.system import TwoTank
    from regelum.objective import RunningObjective
    from regelum.simulator import CasADi

    sampling_time = 0.1
    simulator = CasADi(
        system=TwoTank(),
        state_init=np.array([[2, -2.]]),
        time_final=10,
        max_step=sampling_time/10,
    )
    running_cost = RunningObjective(
        model=ModelQuadLin("diagonal", weights=[10., 10., 0.])
    )


Three-wheeled robot
===================

Description
-----------

A pair of actuated wheels and a ball-wheel are attached to a platform that moves on a flat surface. The wheels roll without
slipping. The pair of control signals for the respective actuators is decomposed into forward velocity U t 1 that is aligned with
the direction in which the robot is facing and angular velocity U t 2 applied to the center of mass of the robot and directed
perpendicular to the platform. The goal is to park the robot at the origin and facing the negative X axis. 

.. math::

    \begin{aligned}
        &\dot{x}_{с} = v \cos(\vartheta), \\
        &\dot{y}_{c} = v \sin(\vartheta), \\
        &\dot{\vartheta} = \omega,
    \end{aligned}

**Variables**

- :math:`x_с` : x-coordinate of the robot [m]
- :math:`y_с` : y-coordinate of the robot [m]
- :math:`\vartheta` : turning angle [rad]
- :math:`v` : speed [m/s]
- :math:`\omega` : revolution speed [rad/s]

**State:** :math:`x_с`, :math:`y_с`, :math:`\vartheta`

**Action:**: :math:`v`, :math:`\omega`

**Action bounds:** :math:`v \in [-25, 25], \\ \omega \in [-5, 5]`

Default configuration of simulator
----------------------------------

- The duration of one episode is set to 5 seconds
- The sampling rate of the pipeline is set to 100 Hz
- Initial state is set to 

.. math::

    \begin{aligned}
        & x_с(0) = 5, \\
        & y_c(0) = 5, \\
        & \vartheta(0) = \frac{3 \pi}{4},
    \end{aligned}


Default running cost
--------------------

.. math::
    
    c(x_с, y_с, \vartheta, v, \omega) = x_c ^ 2 + 10 y_c ^ 2 + \vartheta ^ 2


Usage 
-----

.. code-block:: python

    import numpy as np    
    from regelum.model import ModelQuadLin
    from regelum.system import ThreeWheeledRobotNI
    from regelum.objective import RunningObjective
    from regelum.simulator import CasADi

    sampling_time = 0.01
    simulator = CasADi(
        system=ThreeWheeledRobotNI(),
        state_init=np.array([[5., 5., 3 * np.pi / 4.]]),
        time_final=5,
        max_step=sampling_time / 10.,
    )
    running_cost = RunningObjective(
        model=ModelQuadLin("diagonal", weights=[1., 10., 1., 0., 0.])
    )


Three-wheeled robot with dynamic actuators
==========================================

Description
-----------

Three-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_]

.. math::
    \begin{array}{ll}
        \dot x_с & = v \cos\vartheta \newline
        \dot y_с & = v \sin \vartheta \newline
        \dot \vartheta & = \omega \newline
        \dot v & = \left( \frac 1 m F + q_1 \right) \newline
        \dot \omega & = \left( \frac 1 I M + q_2 \right)
    \end{array}

**Variables** 

- :math:`x_с` : x-coordinate of the robot [m]
- :math:`y_с` : y-coordinate of the robot [m]
- :math:`\vartheta` : turning angle [rad]
- :math:`v` : speed [m/s]
- :math:`\omega` : revolution speed [rad/s]
- :math:`F` : pushing force [N]
- :math:`M` : steering torque [Nm]
- :math:`m` : robot mass [kg]
- :math:`I` : robot moment of inertia around vertical axis [kg m\ :sup:`2`]

**State:** :math:`x_с`, :math:`y_с`, :math:`\vartheta`, :math:`v`, :math:`\omega`

**Action:**: :math:`F`, :math:`M`

**Action bounds:** :math:`v \in [-300, 300], \\ \omega \in [-100, 100]`

Default configuration of simulator
----------------------------------

- The duration of one episode is set to 10 seconds
- The sampling rate of the pipeline is set to 100 Hz
- Initial state is set to 

.. math::

    \begin{aligned}
        & x_с(0) = -5, \\
        & y_c(0) = 5, \\
        & \vartheta(0) = \frac{3 \pi}{4}, \\
        & 0 \\
        & 0
    \end{aligned}

Default running cost
--------------------

.. math::
    
    c(x_с, y_с, \vartheta, v, \omega) = 10 x_c ^ 2 + 10 y_c ^ 2 + \vartheta ^ 2

Usage 
-----

.. code-block:: python

    import numpy as np    
    from regelum.model import ModelQuadLin
    from regelum.system import ThreeWheeledRobot
    from regelum.objective import RunningObjective
    from regelum.simulator import CasADi

    sampling_time = 0.01
    simulator = CasADi(
        system=ThreeWheeledRobot(),
        state_init=np.array([[5., 5., 3 * np.pi / 4., 0., 0.]]),
        time_final=10,
        max_step=sampling_time / 10.,
    )
    running_cost = RunningObjective(
        model=ModelQuadLin("diagonal", weights=[10., 10., 1., 0., 0., 0., 0.])
    )


Cartpole
========

Description
-----------

A pole is attached by an unactuated joint to a cart, which moves along a frictionless track. The goal is to balance the pole by
applying a positive or a negative force :math:`A_t` to the left side of the cart. Below :math:`l`, :math:`m_c` , :math:`m_p` , :math:`g` respectively denote the length of the
pole, the mass of the cart, the mass of the pole and the acceleration of gravity (in SI). The duration of one episode is set to 30
seconds, while the sampling rate of the pipeline is set to 100 Hz.

.. math::
    \begin{array}{ll}
        \dot \theta & = \omega \\
        \dot x & = v_x \\
        \dot \omega & = g \sin{\theta} - \cos{\theta}\frac{A_t + m_p l \omega^2 \sin{\theta}}{\frac{4l}{3}(m_c + m_p) - lm_p \cos{\theta}^2} \\
        \dot v_x & = \frac{A_t + m_p l (\omega ^2 \sin{\theta} - \omega \cos{\theta})}{m_c + m_p}
    \end{array}


**Variables** 

- :math:`\theta` : pole turning angle [rad]
- :math:`x` : x-coordinate of the cart [m]
- :math:`\omega` : pole angular speed with respect to relative coordinate axes with cart in the origin [rad/s]
- :math:`v_x` : absolute speed of the cart [m/s]
- :math:`A_t` : pushing force [N]
- :math:`m_c` : mass of the cart [kg]
- :math:`m_c` : mass of the pole [kg]
- :math:`l` : pole length [m]
  
**State:** :math:`\theta`, :math:`x`, :math:`\omega`, :math:`v_x`

**Action:**: :math:`A_t`

**Action bounds:** :math:`A_t \in [-50, 50]`

Default configuration of simulator
----------------------------------

- The duration of one episode is set to 5 seconds
- The sampling rate of the pipeline is set to 100 Hz
- Initial state is set to 

.. math::

    \begin{aligned}
        & \theta(0) = \pi, \\
        & x(0) = 0, \\
        & \omega(0) = 0, \\
        & v_x = 0, \\
    \end{aligned}

Default running cost
--------------------

.. math::
    
    c(\theta, x, \omega, v_x) = 10\theta ^ 2 + 10 x_c ^ 2 + \omega ^ 2


Usage 
-----

.. code-block:: python

    import numpy as np    
    from regelum.model import ModelQuadLin
    from regelum.system import CartPole
    from regelum.objective import RunningObjective
    from regelum.simulator import CasADi

    sampling_time = 0.01
    simulator = CasADi(
        system=ThreeWheeledRobotNI(),
        state_init=np.array([[np.pi, 0., 0., 0.]]),
        time_final=5,
        max_step=sampling_time / 10.,
    )
    running_cost = RunningObjective(
        model=ModelQuadLin("diagonal", weights=[10., 10., 1., 0.])
    )

Lunar lander
============

Description
-----------

A jet-powered spaceship is approaching the surface of the moon. It can activate its two engines and thus accelerate itself in the
direction opposite to which the activated engine is facing. The goal is to land at the desired location at the appropriate speed
and angle. Below :math:`m`, :math:`J`, :math:`g` respectively denote the mass of the spaceship, the moment of inertia of the spaceship with respect
to its axis of rotation and acceleration of gravity. The duration of one episode is set to 1.5 seconds, while the sampling rate of
the pipeline is set to 100 Hz.

.. math::
    \mathrm{d}\left(\begin{array}{c}x \\ y \\ \theta \\ v_x \\ v_y \\ \omega\end{array}\right)=\left(\begin{array}{c}v_x \\ v_y \\ \omega \\ \frac{1}{m}\left(A_t^1 \cos \theta-A_t^2 \sin \theta\right) \\ \frac{1}{m}\left(A_t^1 \sin \theta+A_t^2 \cos \theta\right)-g \\ \frac{4 A_t^1}{J}\end{array}\right) \mathrm{d} t


**Variables** 

- :math:`x` : x-coordinate of the lander [m]
- :math:`y` : y-coordinate of the lander [m]
- :math:`\theta` : lander rotating angle [rad]
- :math:`v_x` : absolute x-coordinate speed of the lander's center of mass [m/s]
- :math:`v_y` : absolute y-coordinate speed of the lander's center of mass [m/s]
- :math:`\omega` : lander's rotating speed [rad/s]
- :math:`A^1_t` : side pushing force (in egocentric coordinates) [N]
- :math:`A^2_t` :  vertical pushing force (in egocentric coordinates) [N]
- :math:`m` : mass of the lander [kg]
- :math:`g` : acceleration of gravity [m/s^2]
- :math:`J` : moment of inertia of the lander with respect to its axis of rotation [kg m^2]
  
**State:** :math:`x`, :math:`y`, :math:`\theta`, :math:`v_x`, :math:`v_y`, :math:`\omega`

**Action:**: :math:`A^1_t`, :math:`A^2_t`

**Action bounds:** :math:`A^1_t \in [-50, 50]`, :math:`A^2_t \in [-100, 100]`

Default configuration of simulator
----------------------------------

- The duration of one episode is set to 1.5 seconds
- The sampling rate of the pipeline is set to 100 Hz
- Initial state is set to 

.. math::

    \begin{aligned}
        x(0) &= 2, \\
        y(0) &= 5, \\
        \theta(0) &= \pi/3, \\
        v_x(0) &= 0, \\
        v_y(0) &= 0, \\
        \omega(0) &= 0
    \end{aligned}

Default running cost
--------------------

.. math::
    
    c(\theta, x, y, v_x, v_y) = 3 \theta^2 + 5x^2 + 3(y - 1)^2

Usage 
-----

.. code-block:: python

    import numpy as np    
    from regelum.model import ModelQuadLin
    from regelum.system import LunarLander
    from regelum.objective import RunningObjective
    from regelum.simulator import CasADi

    sampling_time = 0.01
    simulator = CasADi(
        system=LunarLander(),
        state_init=np.array([2., 5., np.pi/3., 0., 0., 0.]),
        time_final=1.5,
        max_step=sampling_time / 10.,
    )
    running_cost = RunningObjective(
        model=ModelQuadLin("diagonal", weights=[3., 5., 3.])
    )

Kinematic Point
===============

Description
-----------

A massless point moves on a plane in the direction pointed to by :math:`A_t` with speed :math:`\left\|A_t\right\|`. The goal is to drive the point to the origin. The duration of one episode is set to 5 seconds, while the sampling rate of the pipeline is set to :math:`100 \mathrm{~Hz}`.

.. math::

    \mathrm{d} S_t=\mathrm{d}\left(\begin{array}{l}
    x \\
    y
    \end{array}\right)=\left(\begin{array}{l}
    A_t^1 \\
    A_t^2
    \end{array}\right) \mathrm{d} t

**Variables**

- :math:`x` : x-coordinate of the point [m]
- :math:`y` : y-coordinate of the point [m]
- :math:`A^1_t` : x-directed pushing force [N]
- :math:`A^2_t` : y-directed pushing force [N]

**State:** :math:`x`, :math:`y`

**Action:**: :math:`A^1_t`, :math:`A^2_t`
**Action bounds:** :math:`A^1_t \in [-10, 10]`, :math:`A^2_t \in [-10, 10]`


Default configuration of simulator
----------------------------------

- The duration of one episode is set to 10 seconds
- The sampling rate of the pipeline is set to 100 Hz
- Initial state is set to 

.. math::

    \begin{aligned}
        x(0) &= -10., \\
        y(0) &= -10.
    \end{aligned}

Default running cost
--------------------

.. math::
    
    c(x, y, A^1, A^2) = 10 \left(x^2 + y^2\right)


Usage 
-----

.. code-block:: python

    import numpy as np    
    from regelum.model import ModelQuadLin
    from regelum.system import KinematicPoint
    from regelum.objective import RunningObjective
    from regelum.simulator import CasADi

    sampling_time = 0.01
    simulator = CasADi(
        system=KinematicPoint(),
        state_init=np.array([[2, -2.]]),
        time_final=10,
        max_step=sampling_time/10,
    )
    running_cost = RunningObjective(
        model=ModelQuadLin("diagonal", weights=[10., 10., 0., 0.])
    )