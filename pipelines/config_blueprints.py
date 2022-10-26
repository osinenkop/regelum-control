import numpy as np
import argparse
from abc import abstractmethod
import pickle5 as pickle


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


class RcognitaArgParser(argparse.ArgumentParser):
    def __init__(self, description):

        super().__init__(description=description)
        self.add_argument(
            "--control_mode",
            metavar="control_mode",
            type=str,
            choices=["manual", "nominal", "MPC", "RQL", "SQL", "STAG"],
            default="MPC",
            help="Control mode. Currently available: "
            + "----manual: manual constant control specified by action_manual; "
            + "----nominal: nominal controller, usually used to benchmark optimal controllers;"
            + "----MPC:model-predictive control; "
            + "----RQL: Q-learning actor-critic with prediction_horizon-1 roll-outs of stage objective; "
            + "----SQL: stacked Q-learning; "
            + "----STAG: joint actor-critic (stabilizing), system-specific, needs proper setup.",
        )
        self.add_argument(
            "--is_log",
            action="store_true",
            help="Flag to log data into a data file. Data are stored in simdata folder.",
        )
        self.add_argument(
            "--no_visual",
            action="store_true",
            help="Flag to produce graphical output.",
        )
        self.add_argument(
            "--no_print",
            action="store_true",
            help="Flag to print simulation data into terminal.",
        )
        self.add_argument(
            "--is_est_model",
            action="store_true",
            help="Flag to estimate environment model.",
        )
        self.add_argument(
            "--save_trajectory",
            action="store_true",
            help="Flag to store trajectory inside the pipeline during execution.",
        )
        self.add_argument(
            "--dt",
            type=float,
            metavar="sampling_time",
            dest="sampling_time",
            default=0.01,
            help="Controller sampling time.",
        )
        self.add_argument(
            "--t1",
            type=float,
            metavar="time_final",
            dest="time_final",
            default=10.0,
            help="Final time of episode.",
        )
        self.add_argument("--config", type=open, action=LoadFromFile)
        self.add_argument(
            "strings", metavar="STRING", nargs="*", help="String for searching",
        )

        self.add_argument(
            "-f",
            "--file",
            help="Path for input file. First line should contain number of lines to search in",
        )


class MetaConf(type):
    def __init__(cls, name, bases, clsdict):
        if "argument_parser" in clsdict:

            def new_argument_parser(self):
                args = clsdict["argument_parser"](self)
                self.__dict__.update(vars(args))

            setattr(cls, "argument_parser", new_argument_parser)


class AbstractConfig(object, metaclass=MetaConf):
    @abstractmethod
    def __init__(self):
        self.config_name = None

    @abstractmethod
    def argument_parser(self):
        pass

    @abstractmethod
    def pre_processing(self):
        pass

    def get_env(self):
        self.argument_parser()
        self.pre_processing()
        return self.__dict__

    def config_to_pickle(self):
        with open(
            f"../tests/refs/env_{self.config_name}.pickle", "wb"
        ) as env_description_out:
            pickle.dump(self.__dict__, env_description_out)


class ConfigGridWorld(AbstractConfig):
    def __init__(self):
        self.config_name = "grid_world"

    def argument_parser(self):
        description = "Agent-environment pipeline: grid world."
        parser = RcognitaArgParser(description=description)

        parser.add_argument(
            "--reward_cell",
            type=int,
            nargs="+",
            default=[4, 6],
            action="store",
            help="Coordinates of a reward_cell.",
        )

        parser.add_argument(
            "--grid_size",
            type=int,
            nargs="+",
            default=[9, 9],
            action="store",
            help="Grid world size.",
        )

        parser.add_argument(
            "--discount_factor", type=float, default=1.0, help="Discount factor."
        )
        args = parser.parse_args()
        return args


class Config3WRobot(AbstractConfig):
    def __init__(self):
        self.config_name = "3wrobot"

    def argument_parser(self):
        description = (
            "Agent-environment pipeline: 3-wheel robot with dynamical actuators."
        )

        parser = RcognitaArgParser(description=description)

        parser.add_argument(
            "--Nruns",
            type=int,
            default=1,
            help="Number of episodes. Learned parameters are not reset after an episode.",
        )
        parser.add_argument(
            "--state_init",
            type=str,
            nargs="+",
            metavar="state_init",
            default=["5", "5", "-3*pi/4", "0", "0"],
            help="Initial state (as sequence of numbers); "
            + "dimension is environment-specific!",
        )

        parser.add_argument(
            "--model_est_stage",
            type=float,
            default=1.0,
            help="Seconds to learn model until benchmarking controller kicks in.",
        )
        parser.add_argument(
            "--model_est_period_multiplier",
            type=float,
            default=1,
            help="Model is updated every model_est_period_multiplier times sampling_time seconds.",
        )
        parser.add_argument(
            "--model_order",
            type=int,
            default=5,
            help="Order of state-space estimation model.",
        )
        parser.add_argument(
            "--prob_noise_pow",
            type=float,
            default=False,
            help="Power of probing (exploration) noise.",
        )
        parser.add_argument(
            "--action_manual",
            type=float,
            default=[-5, -3],
            nargs="+",
            help="Manual control action to be fed constant, system-specific!",
        )
        parser.add_argument(
            "--prediction_horizon",
            type=int,
            default=5,
            help="Horizon length (in steps) for predictive controllers.",
        )
        parser.add_argument(
            "--pred_step_size_multiplier",
            type=float,
            default=2.0,
            help="Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time sampling_time.",
        )
        parser.add_argument(
            "--data_buffer_size",
            type=int,
            default=10,
            help="Size of the buffer (experience replay) for model estimation, agent learning etc.",
        )
        parser.add_argument(
            "--running_obj_struct",
            type=str,
            default="quadratic",
            choices=["quadratic", "biquadratic"],
            help="Structure of stage objective function.",
        )
        parser.add_argument(
            "--R1_diag",
            type=float,
            nargs="+",
            default=[1, 10, 1, 0, 0, 0, 0],
            help="Parameter of stage objective function. Must have proper dimension. "
            + "Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--R2_diag",
            type=float,
            nargs="+",
            default=[1, 10, 1, 0, 0, 0, 0],
            help="Parameter of stage objective function . Must have proper dimension. "
            + "Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, "
            + "where diag() is transformation of a vector to a diagonal matrix.",
        )

        parser.add_argument(
            "--discount_factor", type=float, default=1.0, help="Discount factor."
        )
        parser.add_argument(
            "--critic_period_multiplier",
            type=float,
            default=1.0,
            help="Critic is updated every critic_period_multiplier times sampling_time seconds.",
        )
        parser.add_argument(
            "--critic_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix", "quad-mix"],
            help="Feature structure (critic). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms; "
            + "----quad-mix: quadratic, mixed observation-action terms (for, say, action_objective or advantage function approximations).",
        )
        parser.add_argument(
            "--actor_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix"],
            help="Feature structure (actor). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms.",
        )

        args = parser.parse_args()
        return args

    def pre_processing(self):
        self.trajectory = []
        self.dim_state = 5
        self.dim_input = 2
        self.dim_output = self.dim_state
        self.dim_disturb = 0

        self.dim_R1 = self.dim_output + self.dim_input
        self.dim_R2 = self.dim_R1

        # ----------------------------------------Post-processing of arguments
        # Convert `pi` to a number pi
        for k in range(len(self.state_init)):
            self.state_init[k] = eval(self.state_init[k].replace("pi", str(np.pi)))

        self.state_init = np.array(self.state_init)

        self.action_manual = np.array(self.action_manual)

        self.pred_step_size = self.sampling_time * self.pred_step_size_multiplier
        self.model_est_period = self.sampling_time * self.model_est_period_multiplier
        self.critic_period = self.sampling_time * self.critic_period_multiplier
        if self.control_mode == "STAG":
            self.prediction_horizon = 1

        self.R1 = np.diag(np.array(self.R1_diag))
        self.R2 = np.diag(np.array(self.R2_diag))
        self.is_disturb = 0

        self.is_dynamic_controller = 0

        self.time_start = 0

        self.action_init = np.ones(self.dim_input)

        # Solver
        self.atol = 1e-5
        self.rtol = 1e-3

        # xy-plane
        self.xMin = -10
        self.xMax = 10
        self.yMin = -10
        self.yMax = 10

        # Model estimator stores models in a stack and recall the best of model_est_checks
        self.model_est_checks = 0

        # Control constraints
        self.Fmin = -300
        self.Fmax = 300
        self.Mmin = -100
        self.Mmax = 100
        self.action_bounds = np.array([[self.Fmin, self.Fmax], [self.Mmin, self.Mmax]])

        # System parameters
        self.m = 10  # [kg]
        self.I = 1  # [kg m^2]
        self.observation_target = []


class Config3WRobotNI(AbstractConfig):
    def __init__(self):
        self.config_name = "3wrobot_NI"

    def argument_parser(self):
        description = "Agent-environment pipeline: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator)."

        parser = RcognitaArgParser(description=description)

        parser.add_argument(
            "--Nruns",
            type=int,
            default=1,
            help="Number of episodes. Learned parameters are not reset after an episode.",
        )
        parser.add_argument(
            "--state_init",
            type=str,
            nargs="+",
            metavar="state_init",
            default=["5", "5", "-3*pi/4"],
            help="Initial state (as sequence of numbers); "
            + "dimension is environment-specific!",
        )
        parser.add_argument(
            "--model_est_stage",
            type=float,
            default=1.0,
            help="Seconds to learn model until benchmarking controller kicks in.",
        )
        parser.add_argument(
            "--model_est_period_multiplier",
            type=float,
            default=1,
            help="Model is updated every model_est_period_multiplier times sampling_time seconds.",
        )
        parser.add_argument(
            "--model_order",
            type=int,
            default=5,
            help="Order of state-space estimation model.",
        )
        parser.add_argument(
            "--prob_noise_pow",
            type=float,
            default=False,
            help="Power of probing (exploration) noise.",
        )
        parser.add_argument(
            "--action_manual",
            type=float,
            default=[-5, -3],
            nargs="+",
            help="Manual control action to be fed constant, system-specific!",
        )
        parser.add_argument(
            "--prediction_horizon",
            type=int,
            default=3,
            help="Horizon length (in steps) for predictive controllers.",
        )
        parser.add_argument(
            "--pred_step_size_multiplier",
            type=float,
            default=1.0,
            help="Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time sampling_time.",
        )
        parser.add_argument(
            "--data_buffer_size",
            type=int,
            default=10,
            help="Size of the buffer (experience replay) for model estimation, agent learning etc.",
        )
        parser.add_argument(
            "--running_obj_struct",
            type=str,
            default="quadratic",
            choices=["quadratic", "biquadratic"],
            help="Structure of stage objective function.",
        )
        parser.add_argument(
            "--R1_diag",
            type=float,
            nargs="+",
            default=[1, 10, 1, 0, 0],
            help="Parameter of stage objective function. Must have proper dimension. "
            + "Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--R2_diag",
            type=float,
            nargs="+",
            default=[1, 10, 1, 0, 0],
            help="Parameter of stage objective function . Must have proper dimension. "
            + "Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, "
            + "where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--discount_factor", type=float, default=1.0, help="Discount factor."
        )
        parser.add_argument(
            "--critic_period_multiplier",
            type=float,
            default=1.0,
            help="Critic is updated every critic_period_multiplier times sampling_time seconds.",
        )
        parser.add_argument(
            "--critic_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix", "quad-mix", "NN"],
            help="Feature structure (critic). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms; "
            + "----quad-mix: quadratic, mixed observation-action terms (for, say, action_objective or advantage function approximations)."
            + "----NN: Torch neural network.",
        )
        parser.add_argument(
            "--actor_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix"],
            help="Feature structure (actor). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms.",
        )

        args = parser.parse_args()
        return args

    def pre_processing(self):
        self.trajectory = []
        self.dim_state = 3
        self.dim_input = 2
        self.dim_output = self.dim_state
        self.dim_disturb = 2
        if self.control_mode == "STAG":
            self.prediction_horizon = 1

        self.dim_R1 = self.dim_output + self.dim_input
        self.dim_R2 = self.dim_R1
        # ----------------------------------------Post-processing of arguments
        # Convert `pi` to a number pi
        for k in range(len(self.state_init)):
            self.state_init[k] = eval(self.state_init[k].replace("pi", str(np.pi)))

        self.state_init = np.array(self.state_init)
        self.action_manual = np.array(self.action_manual)

        self.pred_step_size = self.sampling_time * self.pred_step_size_multiplier
        self.model_est_period = self.sampling_time * self.model_est_period_multiplier
        self.critic_period = self.sampling_time * self.critic_period_multiplier

        self.R1 = np.diag(np.array(self.R1_diag))
        self.R2 = np.diag(np.array(self.R2_diag))

        assert self.time_final > self.sampling_time > 0.0
        assert self.state_init.size == self.dim_state

        # ----------------------------------------(So far) fixed settings
        self.is_disturb = 0
        self.is_dynamic_controller = 0

        self.time_start = 0

        self.action_init = 0 * np.ones(self.dim_input)

        # Solver
        self.atol = 1e-5
        self.rtol = 1e-3

        # xy-plane
        self.xMin = -10
        self.xMax = 10
        self.yMin = -10
        self.yMax = 10

        # Model estimator stores models in a stack and recall the best of model_est_checks
        self.model_est_checks = 0

        # Control constraints
        self.v_min = -25
        self.v_max = 25
        self.omega_min = -5
        self.omega_max = 5
        self.action_bounds = np.array(
            [[self.v_min, self.v_max], [self.omega_min, self.omega_max]]
        )

        self.xCoord0 = self.state_init[0]
        self.yCoord0 = self.state_init[1]
        self.angle0 = self.state_init[2]
        self.angle_deg_0 = self.angle0 / 2 / np.pi
        self.observation_target = []


class ConfigROS3WRobotNI(Config3WRobotNI):
    def get_env(self):
        self.argument_parser()
        self.pre_processing()
        self.v_min = -0.22
        self.v_max = 0.22
        self.omega_min = -2.84
        self.omega_max = 2.84
        self.action_bounds = np.array(
            [[self.v_min, self.v_max], [self.omega_min, self.omega_max]]
        )
        self.state_init = np.array([2, 2, 3.1415])
        return self.__dict__


class Config2Tank(AbstractConfig):
    def __init__(self):
        self.config_name = "2tank"

    def argument_parser(self):
        description = "Agent-environment pipeline: nonlinear double-tank system."

        parser = RcognitaArgParser(description=description)

        parser.add_argument(
            "--Nruns",
            type=int,
            default=1,
            help="Number of episodes. Learned parameters are not reset after an episode.",
        )
        parser.add_argument(
            "--state_init",
            type=str,
            nargs="+",
            metavar="state_init",
            default=["2", "-2"],
            help="Initial state (as sequence of numbers); "
            + "dimension is environment-specific!",
        )

        parser.add_argument(
            "--model_est_stage",
            type=float,
            default=1.0,
            help="Seconds to learn model until benchmarking controller kicks in.",
        )
        parser.add_argument(
            "--model_est_period_multiplier",
            type=float,
            default=1,
            help="Model is updated every model_est_period_multiplier times sampling_time seconds.",
        )
        parser.add_argument(
            "--model_order",
            type=int,
            default=5,
            help="Order of state-space estimation model.",
        )
        parser.add_argument(
            "--prob_noise_pow",
            type=float,
            default=False,
            help="Power of probing (exploration) noise.",
        )
        parser.add_argument(
            "--action_manual",
            type=float,
            default=[0.5],
            nargs="+",
            help="Manual control action to be fed constant, system-specific!",
        )
        parser.add_argument(
            "--prediction_horizon",
            type=int,
            default=10,
            help="Horizon length (in steps) for predictive controllers.",
        )
        parser.add_argument(
            "--pred_step_size_multiplier",
            type=float,
            default=2.0,
            help="Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time sampling_time.",
        )
        parser.add_argument(
            "--data_buffer_size",
            type=int,
            default=10,
            help="Size of the buffer (experience replay) for model estimation, agent learning etc.",
        )
        parser.add_argument(
            "--running_obj_struct",
            type=str,
            default="quadratic",
            choices=["quadratic", "biquadratic"],
            help="Structure of stage objective function.",
        )
        parser.add_argument(
            "--R1_diag",
            type=float,
            nargs="+",
            default=[10, 10, 1],
            help="Parameter of stage objective function. Must have proper dimension. "
            + "Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--R2_diag",
            type=float,
            nargs="+",
            default=[10, 10, 1],
            help="Parameter of stage objective function . Must have proper dimension. "
            + "Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, "
            + "where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--discount_factor", type=float, default=1.0, help="Discount factor."
        )
        parser.add_argument(
            "--critic_period_multiplier",
            type=float,
            default=1.0,
            help="Critic is updated every critic_period_multiplier times sampling_time seconds.",
        )
        parser.add_argument(
            "--critic_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix", "quad-mix"],
            help="Feature structure (critic). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms; "
            + "----quad-mix: quadratic, mixed observation-action terms (for, say, action_objective or advantage function approximations).",
        )
        parser.add_argument(
            "--actor_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix"],
            help="Feature structure (actor). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms.",
        )

        args = parser.parse_args()
        return args

    def pre_processing(self):
        self.trajectory = []
        self.dim_state = 2
        self.dim_input = 1
        self.dim_output = self.dim_state
        self.dim_disturb = 1

        self.dim_R1 = self.dim_output + self.dim_input
        self.dim_R2 = self.dim_R1

        # ----------------------------------------Post-processing of arguments
        # Convert `pi` to a number pi
        for k in range(len(self.state_init)):
            self.state_init[k] = eval(self.state_init[k].replace("pi", str(np.pi)))

        self.state_init = np.array(self.state_init)
        self.action_manual = np.array(self.action_manual)

        self.pred_step_size = self.sampling_time * self.pred_step_size_multiplier
        self.model_est_period = self.sampling_time * self.model_est_period_multiplier
        self.critic_period = self.sampling_time * self.critic_period_multiplier

        self.R1 = np.diag(np.array(self.R1_diag))
        self.R2 = np.diag(np.array(self.R2_diag))

        assert self.time_final > self.sampling_time > 0.0
        assert self.state_init.size == self.dim_state

        # ----------------------------------------(So far) fixed settings
        self.is_disturb = 0
        self.is_dynamic_controller = 0

        self.time_start = 0

        self.action_init = 0.5 * np.ones(self.dim_input)

        self.disturb_init = 0 * np.ones(self.dim_disturb)

        # Solver
        self.atol = 1e-5
        self.rtol = 1e-3

        # Model estimator stores models in a stack and recall the best of model_est_checks
        self.model_est_checks = 0

        # Control constraints
        self.action_min = 0
        self.action_max = 1
        self.action_bounds = np.array([[self.action_min], [self.action_max]]).T

        # System parameters
        self.tau1 = 18.4
        self.tau2 = 24.4
        self.K1 = 1.3
        self.K2 = 1
        self.K3 = 0.2

        # Target filling of the tanks
        self.observation_target = np.array([0.5, 0.5])


class ConfigInvertedPendulum(AbstractConfig):
    def __init__(self):
        self.config_name = "inverted-pendulum"

    def argument_parser(self):
        description = (
            "Agent-environment pipeline: 3-wheel robot with dynamical actuators."
        )

        parser = RcognitaArgParser(description=description)

        parser.add_argument(
            "--is_playback", action="store_true", help="Flag to playback.",
        )

        parser.add_argument(
            "--state_init",
            type=float,
            nargs="+",
            metavar="state_init",
            default=[np.pi, 0.0],
            help="Initial state (as sequence of numbers); "
            + "dimension is environment-specific!",
        )

        parser.add_argument(
            "--model_order",
            type=int,
            default=5,
            help="Order of state-space estimation model.",
        )
        parser.add_argument(
            "--prob_noise_pow",
            type=float,
            default=False,
            help="Power of probing (exploration) noise.",
        )
        parser.add_argument(
            "--sigma",
            type=float,
            default=1,
            help="Power of probing (exploration) noise.",
        )
        parser.add_argument(
            "--prediction_horizon",
            type=int,
            default=5,
            help="Horizon length (in steps) for predictive controllers.",
        )
        parser.add_argument(
            "--N_episodes",
            type=int,
            default=1,
            help="Number of episodes in one iteration",
        )
        parser.add_argument(
            "--N_iterations",
            type=int,
            default=1,
            help="Number of iterations in episodical scenario",
        )
        parser.add_argument(
            "--pred_step_size_multiplier",
            type=float,
            default=2.0,
            help="Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time sampling_time.",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.01,
            help="Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time sampling_time.",
        )
        parser.add_argument(
            "--learning_rate_critic",
            type=float,
            default=0.001,
            help="Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time sampling_time.",
        )
        parser.add_argument(
            "--speedup", type=int, default=20, help="Animation speed up",
        )
        parser.add_argument(
            "--data_buffer_size",
            type=int,
            default=10,
            help="Size of the buffer (experience replay) for model estimation, agent learning etc.",
        )
        parser.add_argument(
            "--running_obj_struct",
            type=str,
            default="quadratic",
            choices=["quadratic", "biquadratic"],
            help="Structure of stage objective function.",
        )
        parser.add_argument(
            "--R1_diag",
            type=float,
            nargs="+",
            default=[-10, 0, -3, -1],
            help="Parameter of stage objective function. Must have proper dimension. "
            + "Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--R2_diag",
            type=float,
            nargs="+",
            default=[-10, 0, -3, 0],
            help="Parameter of stage objective function . Must have proper dimension. "
            + "Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, "
            + "where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--initial_weights",
            type=float,
            nargs="+",
            default=np.array([30, 0.0, 9]),
            help="Parameters of the gaussian model for prior mean computation",
        )
        parser.add_argument(
            "--discount_factor", type=float, default=1.0, help="Discount factor."
        )
        parser.add_argument(
            "--critic_period_multiplier",
            type=float,
            default=1.0,
            help="Critic is updated every critic_period_multiplier times sampling_time seconds.",
        )
        parser.add_argument(
            "--critic_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix", "quad-mix"],
            help="Feature structure (critic). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms; "
            + "----quad-mix: quadratic, mixed observation-action terms (for, say, action_objective or advantage function approximations).",
        )
        parser.add_argument(
            "--is_fixed_critic_weights",
            action="store_true",
            help="Flag to fix critic parameters",
        )
        parser.add_argument(
            "--fixed_actor_parameters",
            action="store_true",
            help="Flag to fix actor parameters",
        )
        args = parser.parse_args()
        return args

    def pre_processing(self):
        self.trajectory = []
        self.dim_state = 2
        self.dim_input = 1
        self.dim_output = self.dim_state + 1
        self.dim_disturb = 0

        self.dim_R1 = self.dim_output + self.dim_input
        self.dim_R2 = self.dim_R1

        # ----------------------------------------Post-processing of arguments
        # Convert `pi` to a number pi

        # self.state_init = [eval(self.state_init[k].replace("pi", str(np.pi)))]

        self.state_init = np.array(self.state_init)

        self.pred_step_size = self.sampling_time * self.pred_step_size_multiplier
        self.critic_period = self.sampling_time * self.critic_period_multiplier
        if self.control_mode == "STAG":
            self.prediction_horizon = 1

        self.R1 = np.diag(np.array(self.R1_diag))
        self.R2 = np.diag(np.array(self.R2_diag))
        self.is_disturb = 0

        self.is_dynamic_controller = 0

        self.time_start = 0

        self.action_init = np.ones(self.dim_input)

        # Solver
        self.atol = 1e-5
        self.rtol = 1e-3

        # Control constraints
        self.Mmin = -30
        self.Mmax = 30
        self.action_bounds = np.array([[self.Mmin, self.Mmax]])

        # System parameters
        self.m = 1  # [kg]
        self.g = 9.81  # [kg m^2]
        self.l = 1
        self.observation_target = []


class ConfigInvertedPendulumAC(AbstractConfig):
    def __init__(self):
        self.config_name = "inverted-pendulum-AC"

    def argument_parser(self):
        description = (
            "Agent-environment pipeline: 3-wheel robot with dynamical actuators."
        )

        parser = RcognitaArgParser(description=description)

        parser.add_argument(
            "--is_playback", action="store_true", help="Flag to playback.",
        )

        parser.add_argument(
            "--t1_critic",
            type=float,
            metavar="time_final_critic",
            dest="time_final_critic",
            default=1.0,
            help="Final time of critic episode.",
        )

        parser.add_argument(
            "--state_init",
            type=float,
            nargs="+",
            metavar="state_init",
            default=[np.pi, 0.0],
            help="Initial state (as sequence of numbers); "
            + "dimension is environment-specific!",
        )

        parser.add_argument(
            "--model_order",
            type=int,
            default=5,
            help="Order of state-space estimation model.",
        )
        parser.add_argument(
            "--prob_noise_pow",
            type=float,
            default=False,
            help="Power of probing (exploration) noise.",
        )
        parser.add_argument(
            "--sigma",
            type=float,
            default=1,
            help="Power of probing (exploration) noise.",
        )
        parser.add_argument(
            "--prediction_horizon",
            type=int,
            default=5,
            help="Horizon length (in steps) for predictive controllers.",
        )
        parser.add_argument(
            "--N_episodes_critic",
            type=int,
            default=1,
            help="Number of episodes in one critic criteration",
        )
        parser.add_argument(
            "--N_iterations_critic",
            type=int,
            default=2,
            help="Number of iterations in episodical critic scenario",
        )
        parser.add_argument(
            "--N_episodes_actor",
            type=int,
            default=4,
            help="Number of episodes in one actor iteration",
        )
        parser.add_argument(
            "--N_iterations_actor",
            type=int,
            default=10,
            help="Number of iterations in episodical actor scenario",
        )

        parser.add_argument(
            "--pred_step_size_multiplier",
            type=float,
            default=2.0,
            help="Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time sampling_time.",
        )
        parser.add_argument(
            "--learning_rate_actor",
            type=float,
            default=0.01,
            help="Size of PG actor learning rate.",
        )
        parser.add_argument(
            "--learning_rate_critic",
            type=float,
            default=0.0000001,
            help="Size of gradient critic update.",
        )
        parser.add_argument(
            "--speedup", type=int, default=20, help="Animation speed up",
        )
        parser.add_argument(
            "--data_buffer_size",
            type=int,
            default=10,
            help="Size of the buffer (experience replay) for model estimation, agent learning etc.",
        )
        parser.add_argument(
            "--running_obj_struct",
            type=str,
            default="quadratic",
            choices=["quadratic", "biquadratic"],
            help="Structure of stage objective function.",
        )
        parser.add_argument(
            "--R1_diag",
            type=float,
            nargs="+",
            default=[-10, 0, -3, -1],
            help="Parameter of stage objective function. Must have proper dimension. "
            + "Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--R2_diag",
            type=float,
            nargs="+",
            default=[-10, 0, -3, 0],
            help="Parameter of stage objective function . Must have proper dimension. "
            + "Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, "
            + "where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--initial_weights",
            type=float,
            nargs="+",
            default=np.array([30, 0.0, 9]),
            help="Parameters of the gaussian model for prior mean computation",
        )
        parser.add_argument(
            "--discount_factor", type=float, default=1.0, help="Discount factor."
        )
        parser.add_argument(
            "--critic_period_multiplier",
            type=float,
            default=1.0,
            help="Critic is updated every critic_period_multiplier times sampling_time seconds.",
        )
        parser.add_argument(
            "--critic_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix", "quad-mix"],
            help="Feature structure (critic). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms; "
            + "----quad-mix: quadratic, mixed observation-action terms (for, say, action_objective or advantage function approximations).",
        )
        parser.add_argument(
            "--is_fixed_critic_weights",
            action="store_true",
            help="Flag to fix critic parameters",
        )
        parser.add_argument(
            "--fixed_actor_parameters",
            action="store_true",
            help="Flag to fix actor parameters",
        )
        args = parser.parse_args()
        return args

    def pre_processing(self):
        self.trajectory = []
        self.dim_state = 2
        self.dim_input = 1
        self.dim_output = self.dim_state + 1
        self.dim_disturb = 0

        self.dim_R1 = self.dim_output + self.dim_input
        self.dim_R2 = self.dim_R1

        # ----------------------------------------Post-processing of arguments
        # Convert `pi` to a number pi

        # self.state_init = [eval(self.state_init[k].replace("pi", str(np.pi)))]

        self.state_init = np.array(self.state_init)

        self.pred_step_size = self.sampling_time * self.pred_step_size_multiplier
        self.critic_period = self.sampling_time * self.critic_period_multiplier
        if self.control_mode == "STAG":
            self.prediction_horizon = 1

        self.R1 = np.diag(np.array(self.R1_diag))
        self.R2 = np.diag(np.array(self.R2_diag))
        self.is_disturb = 0

        self.is_dynamic_controller = 0

        self.time_start = 0

        self.action_init = np.ones(self.dim_input)

        # Solver
        self.atol = 1e-5
        self.rtol = 1e-3

        # Control constraints
        self.Mmin = -30
        self.Mmax = 30
        self.action_bounds = np.array([[self.Mmin, self.Mmax]])
        self.prediction_horizon = 0

        # System parameters
        self.m = 1  # [kg]
        self.g = 9.81  # [kg m^2]
        self.l = 1
        self.observation_target = []
