from abc import ABC, abstractmethod
from rcognita import controllers, simulator, predictors, optimizers, objectives

from rcognita.utilities import rc
from rcognita.actors import ActorCALF, ActorMPC, ActorRQL, ActorSQL, ActorRPO, ActorCLF

from rcognita.critics import (
    CriticOfActionObservation,
    CriticCALF,
    CriticTrivial,
    CriticOfObservation,
)

from rcognita.models import (
    ModelQuadLin,
    ModelQuadratic,
    ModelQuadNoMix,
    ModelQuadNoMix2D,
    ModelQuadForm,
    ModelSS,
    ModelWeightContainer,
    ModelQuadNoMixTorch,
)

from rcognita.scenarios import EpisodicScenario
import matplotlib.animation as animation
from rcognita.utilities import on_key_press
import matplotlib.pyplot as plt


class Pipeline(ABC):
    @property
    @abstractmethod
    def config(self):
        return self.config

    def load_config(self):
        self.env_config = self.config()

    def setup_env(self):
        self.__dict__.update(self.env_config.get_env())
        self.trajectory = []

    def config_to_pickle(self):
        self.env_config.config_to_pickle()

    @abstractmethod
    def initialize_system(self):
        pass

    @abstractmethod
    def initialize_predictor(self):
        pass

    @abstractmethod
    def initialize_controller(self):
        pass

    @abstractmethod
    def initialize_controller(self):
        pass

    @abstractmethod
    def initialize_simulator(self):
        pass

    @abstractmethod
    def initialize_logger(self):
        pass

    def main_loop_raw(self):
        pass

    @abstractmethod
    def execute_pipeline(self):
        pass


class PipelineWithDefaults(Pipeline):
    def initialize_predictor(self):
        self.predictor = predictors.EulerPredictor(
            self.pred_step_size,
            self.system.compute_dynamics,
            self.system.out,
            self.dim_output,
            self.prediction_horizon,
        )

    # def initialize_predictor(self):
    #     self.predictor = predictors.RKPredictor(
    #         self.state_init,
    #         self.action_init,
    #         self.pred_step_size,
    #         self.system.compute_dynamics,
    #         self.system.out,
    #         self.dim_output,
    #         self.prediction_horizon,
    #     )

    def initialize_models(self):
        if self.control_mode in ("CALF", "AC"):
            self.dim_critic_model_input = self.dim_output
        else:
            self.dim_critic_model_input = self.dim_input + self.dim_output

        if self.critic_struct == "NN":
            self.critic_model = ModelQuadNoMixTorch(
                self.dim_output, self.dim_input, dim_hidden=3
            )
        else:
            if self.critic_struct == "quad-lin":
                self.critic_model = ModelQuadLin(self.dim_critic_model_input)
            elif self.critic_struct == "quad-nomix":
                self.critic_model = ModelQuadNoMix(self.dim_critic_model_input)
            elif self.critic_struct == "quad-nomix-2d":
                self.critic_model = ModelQuadNoMix2D(self.dim_critic_model_input)
            elif self.critic_struct == "quadratic":
                self.critic_model = ModelQuadratic(self.dim_critic_model_input)

        if self.actor_struct == "NN":
            self.critic_model = ModelQuadNoMixTorch(self.dim_output, dim_hidden=3)
        else:
            if self.actor_struct == "quad-lin":
                self.actor_model = ModelQuadLin(self.dim_output)
            elif self.actor_struct == "quad-nomix":
                self.actor_model = ModelQuadNoMix(self.dim_output)
            elif self.actor_struct == "quadratic":
                self.actor_model = ModelQuadratic(self.dim_output)
            else:
                self.actor_model = ModelWeightContainer(
                    dim_output=self.dim_input, weights_init=self.action_init
                )

        self.model_running_objective = ModelQuadForm(weights=self.R1)

    def initialize_objectives(self):

        self.running_objective = objectives.RunningObjective(
            model=self.model_running_objective
        )

    def initialize_optimizers(self):
        opt_options = {
            "maxiter": 100,
            "maxfev": 5000,
            "disp": False,
            "adaptive": True,
            "xatol": 1e-7,
            "fatol": 1e-7,
        }
        self.actor_optimizer = optimizers.SciPyOptimizer(
            opt_method="SLSQP", opt_options=opt_options
        )
        self.critic_optimizer = optimizers.SciPyOptimizer(
            opt_method="SLSQP", opt_options=opt_options,
        )

    def initialize_actor_critic(self):
        if self.control_mode == "LF":
            self.critic = CriticTrivial(self.running_objective, self.sampling_time)

            self.actor = ActorCLF(
                self.nominal_controller,
                self.prediction_horizon,
                self.dim_input,
                self.dim_output,
                self.control_mode,
                action_bounds=self.action_bounds,
                action_init=self.action_init,
                predictor=self.predictor,
                optimizer=self.actor_optimizer,
                critic=self.critic,
                running_objective=self.running_objective,
                model=self.actor_model,
            )

        elif self.control_mode == "CALF":
            self.critic = CriticCALF(
                safe_decay_rate=self.safe_decay_rate,
                is_dynamic_decay_rate=self.is_dynamic_decay_rate,
                dim_input=self.dim_input,
                dim_output=self.dim_output,
                data_buffer_size=self.data_buffer_size,
                running_objective=self.running_objective,
                discount_factor=self.discount_factor,
                optimizer=self.critic_optimizer,
                model=self.critic_model,
                predictor=self.predictor,
                observation_init=self.state_init,
                safe_controller=self.nominal_controller,
                penalty_param=self.penalty_param,
                sampling_time=self.sampling_time,
                critic_regularization_param=self.critic_regularization_param,
            )
            self.actor = ActorCALF(
                self.nominal_controller,
                self.prediction_horizon,
                self.dim_input,
                self.dim_output,
                self.control_mode,
                action_bounds=self.action_bounds,
                action_init=self.action_init,
                predictor=self.predictor,
                optimizer=self.actor_optimizer,
                critic=self.critic,
                running_objective=self.running_objective,
                model=self.actor_model,
                penalty_param=self.penalty_param,
                actor_regularization_param=self.actor_regularization_param,
            )

        else:
            if self.control_mode == "MPC":
                Actor = ActorMPC
                self.critic = CriticTrivial(self.running_objective, self.sampling_time)
            elif self.control_mode == "AC":
                self.critic = CriticOfObservation(
                    dim_input=self.dim_input,
                    dim_output=self.dim_output,
                    data_buffer_size=self.data_buffer_size,
                    running_objective=self.running_objective,
                    discount_factor=self.discount_factor,
                    optimizer=self.critic_optimizer,
                    model=self.critic_model,
                    sampling_time=self.sampling_time,
                )
                Actor = ActorRPO

            else:
                self.critic = CriticOfActionObservation(
                    dim_input=self.dim_input,
                    dim_output=self.dim_output,
                    data_buffer_size=self.data_buffer_size,
                    running_objective=self.running_objective,
                    discount_factor=self.discount_factor,
                    optimizer=self.critic_optimizer,
                    model=self.critic_model,
                    sampling_time=self.sampling_time,
                )
                if self.control_mode == "RQL":
                    Actor = ActorRQL
                elif self.control_mode == "SQL":
                    Actor = ActorSQL
                elif self.control_mode == "nominal":
                    Actor = ActorMPC

            self.actor = Actor(
                self.prediction_horizon,
                self.dim_input,
                self.dim_output,
                self.control_mode,
                self.action_bounds,
                action_init=self.action_init,
                predictor=self.predictor,
                optimizer=self.actor_optimizer,
                critic=self.critic,
                running_objective=self.running_objective,
                model=self.actor_model,
            )

    def initialize_controller(self):
        if self.control_mode == "nominal":
            self.controller = self.nominal_controller
        elif self.control_mode == "CALF":
            self.controller = controllers.CALFControllerExPost(
                time_start=self.time_start,
                sampling_time=self.sampling_time,
                critic_period=self.critic_period,
                actor=self.actor,
                critic=self.critic,
                observation_target=None,
            )
        else:
            self.controller = controllers.RLController(
                time_start=self.time_start,
                sampling_time=self.sampling_time,
                critic_period=self.critic_period,
                actor=self.actor,
                critic=self.critic,
                observation_target=None,
            )

    def initialize_simulator(self):
        self.simulator = simulator.Simulator(
            system=self.system,
            sys_type="diff_eqn",
            state_init=self.state_init,
            disturb_init=None,
            action_init=self.action_init,
            time_start=self.time_start,
            time_final=self.time_final,
            sampling_time=self.sampling_time,
            max_step=self.sampling_time / 10.0,
            first_step=1e-6,
            atol=self.atol,
            rtol=self.rtol,
            is_disturb=self.is_disturb,
            is_dynamic_controller=self.is_dynamic_controller,
            ode_backend=self.ode_backend,
        )

    def initialize_scenario(self):
        self.scenario = EpisodicScenario(
            1,
            1,
            self.system,
            self.simulator,
            self.controller,
            self.actor,
            self.critic,
            self.logger,
            self.datafiles,
            self.time_final,
            self.running_objective,
            no_print=self.no_print,
            is_log=self.is_log,
            is_playback=self.is_playback,
            state_init=self.state_init,
            action_init=self.action_init,
            speedup=self.speedup,
        )

    def main_loop_visual(self):
        self.animator.init_anim()

        anm = animation.FuncAnimation(
            self.animator.main_figure,
            self.animator.animate,
            blit=True,
            interval=self.sampling_time / 1e6,
            repeat=False,
        )

        self.animator.get_anm(anm)

        cId = self.animator.main_figure.canvas.mpl_connect(
            "key_press_event", lambda event: on_key_press(event, anm)
        )

        anm.running = True

        self.animator.main_figure.tight_layout()

        plt.show()

    def execute_pipeline(self, **kwargs):
        self.load_config()
        self.setup_env()
        self.__dict__.update(kwargs)
        self.initialize_system()
        self.initialize_predictor()
        self.initialize_nominal_controller()
        self.initialize_models()
        self.initialize_objectives()
        self.initialize_optimizers()
        self.initialize_actor_critic()
        self.initialize_controller()
        self.initialize_simulator()
        self.initialize_logger()
        self.initialize_scenario()
        if not self.no_visual and not self.save_trajectory:
            self.initialize_visualizer()
            self.main_loop_visual()
        else:
            self.scenario.run()
            if self.is_playback:
                self.initialize_visualizer()
                self.main_loop_visual()
