import os, sys

from rcognita.visualization import animator

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

import pathlib
import warnings
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import csv
import rcognita

from config_blueprints import Config2Tank
from pipeline_blueprints import Pipeline

if os.path.abspath(rcognita.__file__ + "/../..") == PARENT_DIR:
    info = (
        f"this script is being run using "
        f"rcognita ({rcognita.__version__}) "
        f"located in cloned repository at '{PARENT_DIR}'. "
        f"If you are willing to use your locally installed rcognita, "
        f"run this script ('{os.path.basename(__file__)}') outside "
        f"'rcognita/presets'."
    )
else:
    info = (
        f"this script is being run using "
        f"locally installed rcognita ({rcognita.__version__}). "
        f"Make sure the versions match."
    )
print("INFO:", info)

from rcognita import (
    controllers,
    simulator,
    systems,
    loggers,
    predictors,
    optimizers,
    models,
    objectives,
)
from datetime import datetime
from rcognita.utilities import on_key_press
from rcognita.actors import (
    ActorCALF,
    ActorMPC,
    ActorRQL,
    ActorSQL,
)

from rcognita.critics import (
    CriticOfActionObservation,
    CriticCALF,
)


class Pipeline2Tank(Pipeline):
    def initialize_system(self):
        self.system = systems.System2Tank(
            sys_type="diff_eqn",
            dim_state=self.dim_state,
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            dim_disturb=self.dim_disturb,
            pars=[self.tau1, self.tau2, self.K1, self.K2, self.K3],
            action_bounds=self.action_bounds,
        )

    def initialize_state_predictor(self):
        self.predictor = predictors.EulerPredictor(
            self.pred_step_size,
            self.system._state_dyn,
            self.system.out,
            self.dim_output,
            self.prediction_horizon,
        )

    def initialize_objectives(self):
        self.objective = objectives.RunningObjective(
            model=models.ModelQuadForm(R1=self.R1, R2=self.R2)
        )

    def initialize_optimizers(self):
        opt_options = {
            "maxiter": 200,
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
        self.nominal_controller = controllers.Controller3WRobotNIDisassembledCLF(
            controller_gain=0.5,
            action_bounds=self.action_bounds,
            time_start=self.time_start,
            sampling_time=self.sampling_time,
        )

        if self.control_mode == "CALF":
            Critic = CriticCALF
            Actor = ActorCALF

        else:
            Critic = CriticOfActionObservation
            if self.control_mode == "MPC":
                Actor = ActorMPC
            elif self.control_mode == "RQL":
                Actor = ActorRQL
            elif self.control_mode == "SQL":
                Actor = ActorSQL

        self.critic = Critic(
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            data_buffer_size=self.data_buffer_size,
            running_objective=self.running_objective,
            discount_factor=self.discount_factor,
            optimizer=self.critic_optimizer,
            model=models.ModelPolynomial(model_name=self.critic_struct),
            safe_controller=self.nominal_controller,
            predictor=self.predictor,
        )
        self.actor = Actor(
            self.prediction_horizon,
            self.dim_input,
            self.dim_output,
            self.control_mode,
            self.action_bounds,
            predictor=self.predictor,
            optimizer=self.actor_optimizer,
            critic=self.critic,
            running_objective=self.running_objective,
        )

    def initialize_controller(self):
        self.controller = controllers.RLController(
            action_init=self.action_init,
            time_start=self.time_start,
            sampling_time=self.sampling_time,
            pred_step_size=self.pred_step_size,
            compute_state_dynamics=self.system._state_dyn,
            sys_out=self.system.out,
            model_est_stage=self.model_est_stage,
            model_est_period=self.model_est_period,
            data_buffer_size=self.data_buffer_size,
            model_est_checks=self.model_est_checks,
            critic_period=self.critic_period,
            actor=self.actor,
            critic=self.critic,
            running_obj_pars=[self.R1],
            observation_target=self.observation_target,
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
            max_step=self.sampling_time / 2,
            first_step=1e-6,
            atol=self.atol,
            rtol=self.rtol,
            is_disturb=self.is_disturb,
            is_dynamic_controller=self.is_dynamic_controller,
        )

    def main_loop_visual(self):
        self.state_full_init = self.simulator.state_full

        animator = animator.Animator2Tank(
            objects=(
                self.simulator,
                self.system,
                [],
                self.controller,
                self.datafiles,
                self.logger,
            ),
            pars=(
                self.state_init,
                self.action_init,
                self.time_start,
                self.time_final,
                self.state_full_init,
                self.control_mode,
                self.action_manual,
                self.action_min,
                self.action_max,
                self.no_print,
                self.is_log,
                0,
                [],
                self.observation_target,
            ),
        )

        anm = animation.FuncAnimation(
            animator.fig_sim,
            animator.animate,
            init_func=animator.init_anim,
            blit=False,
            interval=self.sampling_time / 1e6,
            repeat=False,
        )

        animator.get_anm(anm)

        cId = animator.fig_sim.canvas.mpl_connect(
            "key_press_event", lambda event: on_key_press(event, anm)
        )

        anm.running = True

        animator.fig_sim.tight_layout()

        plt.show()

    def execute_pipeline(self, **kwargs):
        self.load_config(Config2Tank)
        self.setup_env()
        self.__dict__.update(kwargs)
        self.initialize_system()
        self.initialize_state_predictor()
        self.initialize_objectives()
        self.initialize_optimizers()
        self.initialize_actor_critic()
        self.initialize_controller()
        self.initialize_simulator()
        self.initialize_logger()
        if not self.no_visual and not self.save_trajectory:
            self.main_loop_visual()
        else:
            self.main_loop_raw()


if __name__ == "__main__":

    Pipeline2Tank().execute_pipeline()
