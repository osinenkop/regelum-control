import os, sys

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
from pipeline_blueprints import AbstractPipeline

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
    animators,
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
    ActorSTAG,
    ActorMPC,
    ActorRQL,
    ActorSQL,
)

from rcognita.critics import (
    CriticActionValue,
    CriticSTAG,
)


class Pipeline2Tank(AbstractPipeline):
    def initialize_system(self):
        self.system = systems.Sys2Tank(
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
            opt_method="SLSQP",
            opt_options=opt_options,
        )

    def initialize_actor_critic(self):
        self.nominal_controller = controllers.NominalController3WRobotNI(
            controller_gain=0.5,
            action_bounds=self.action_bounds,
            time_start=self.time_start,
            sampling_time=self.sampling_time,
        )

        if self.control_mode == "STAG":
            Critic = CriticSTAG
            Actor = ActorSTAG

        else:
            Critic = CriticActionValue
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
            prob_noise_pow=self.prob_noise_pow,
            is_est_model=self.is_est_model,
            model_est_stage=self.model_est_stage,
            model_est_period=self.model_est_period,
            data_buffer_size=self.data_buffer_size,
            model_order=self.model_order,
            model_est_checks=self.model_est_checks,
            critic_period=self.critic_period,
            actor=self.actor,
            critic=self.critic,
            running_obj_pars=[self.R1],
            observation_target=[],
        )

    def initialize_simulator(self):
        self.simulator = simulator.Simulator(
            sys_type="diff_eqn",
            compute_closed_loop_rhs=self.system.compute_closed_loop_rhs,
            sys_out=self.system.out,
            state_init=self.state_init,
            disturb_init=[],
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

    def initialize_logger(self):
        if (
            os.path.basename(os.path.normpath(os.path.abspath(os.getcwd())))
            == "presets"
        ):
            self.data_folder = "../simdata"
        else:
            self.data_folder = "simdata"

        pathlib.Path(self.data_folder).mkdir(parents=True, exist_ok=True)

        date = datetime.now().strftime("%Y-%m-%d")
        time = datetime.now().strftime("%Hh%Mm%Ss")
        self.datafiles = [None] * self.Nruns

        for k in range(0, self.Nruns):
            self.datafiles[k] = (
                self.data_folder
                + "/"
                + self.system.name
                + "__"
                + self.control_mode
                + "__"
                + date
                + "__"
                + time
                + "__run{run:02d}.csv".format(run=k + 1)
            )

            if self.is_log:
                print("Logging data to:    " + self.datafiles[k])

                with open(self.datafiles[k], "w", newline="") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(["System", self.system.name])
                    writer.writerow(["Controller", self.control_mode])
                    writer.writerow(["sampling_time", str(self.sampling_time)])
                    writer.writerow(["state_init", str(self.state_init)])
                    writer.writerow(["is_est_model", str(self.is_est_model)])
                    writer.writerow(["model_est_stage", str(self.model_est_stage)])
                    writer.writerow(
                        [
                            "model_est_period_multiplier",
                            str(self.model_est_period_multiplier),
                        ]
                    )
                    writer.writerow(["model_order", str(self.model_order)])
                    writer.writerow(["prob_noise_pow", str(self.prob_noise_pow)])
                    writer.writerow(
                        ["prediction_horizon", str(self.prediction_horizon)]
                    )
                    writer.writerow(
                        [
                            "pred_step_size_multiplier",
                            str(self.pred_step_size_multiplier),
                        ]
                    )
                    writer.writerow(["data_buffer_size", str(self.data_buffer_size)])
                    writer.writerow(
                        ["running_obj_struct", str(self.running_obj_struct)]
                    )
                    writer.writerow(["R1_diag", str(self.R1_diag)])
                    writer.writerow(["R2_diag", str(self.R2_diag)])
                    writer.writerow(["discount_factor", str(self.discount_factor)])
                    writer.writerow(
                        ["critic_period_multiplier", str(self.critic_period_multiplier)]
                    )
                    writer.writerow(["critic_struct", str(self.critic_struct)])
                    writer.writerow(["actor_struct", str(self.actor_struct)])
                    writer.writerow(
                        ["t [s]", "h1", "h2", "p", "running_objective", "outcome"]
                    )

        # Do not display annoying warnings when print is on
        if not self.no_print:
            warnings.filterwarnings("ignore")

        self.logger = loggers.Logger2Tank()

    def main_loop_visual(self):
        self.state_full_init = self.simulator.state_full

        animator = animators.Animator2Tank(
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
                self.Nruns,
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

    def main_loop_raw(self):
        run_curr = 1
        datafile = self.datafiles[0]

        while True:
            self.simulator.do_sim_step()

            time, state, observation, state_full = self.simulator.get_sim_step_data()

            if self.save_trajectory:
                self.trajectory.append(state_full)

            action = self.controller.compute_action(time, observation)

            self.system.receive_action(action)
            self.controller.update_outcome(observation, action)

            h1 = state_full[0]
            h2 = state_full[1]
            p = action

            running_objective = self.controller.running_objective(observation, action)
            outcome = self.controller.outcome_value

            if not self.no_print:
                self.logger.print_sim_step(time, h1, h2, p, running_objective, outcome)

            if self.is_log:
                self.logger.log_data_row(
                    datafile, time, h1, h2, p, running_objective, outcome
                )

            if time >= self.time_final:
                if not self.no_print:
                    print(
                        ".....................................Run {run:2d} done.....................................".format(
                            run=run_curr
                        )
                    )

                run_curr += 1

                if run_curr > self.Nruns:
                    break

                if self.is_log:
                    datafile = self.datafiles[run_curr - 1]

                # Reset simulator
                self.simulator.status = "running"
                self.simulator.time = self.time_start
                self.simulator.observation = self.state_full_init

                if self.control_mode != "nominal":
                    self.controller.reset(self.time_start)
                else:
                    self.nominal_controller.reset(self.time_start)

                outcome = 0

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
