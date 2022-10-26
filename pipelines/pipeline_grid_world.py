import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import rcognita

from config_blueprints import ConfigGridWorld
from pipeline_blueprints import PipelineWithDefaults

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
    objectives,
    models,
    utilities,
)
from rcognita.loggers import logger3WRobotNI
from datetime import datetime
from rcognita.utilities import on_key_press
from rcognita.actors import ActorTabular

from rcognita.critics import CriticTabularVI

from rcognita.utilities import rc

from rcognita.scenarios import TabularScenarioVI

from enum import IntEnum


class Actions(IntEnum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3
    TURNAROUND = 4


RIGHT = Actions.RIGHT
LEFT = Actions.LEFT
UP = Actions.UP
DOWN = Actions.DOWN
TURNAROUND = Actions.TURNAROUND

action_space = [RIGHT, LEFT, UP, DOWN, TURNAROUND]
action_space_size = len(action_space)
grid_size = [9, 9]
punishment = -1e2


def slice2indices(sl, fixed_x=None, fixed_y=None):
    min_ind = sl.start
    max_ind = sl.stop
    if fixed_x:
        return [[fixed_x, sl.indices(i)[1]] for i in range(min_ind, max_ind)]
    elif fixed_y:
        return [[sl.indices(i)[1], fixed_y] for i in range(min_ind, max_ind)]


class PipelineTabular(PipelineWithDefaults):
    config = ConfigGridWorld

    def initialize_logger(self):
        pass

    def initialize_system(self):
        self.system = systems.GridWorld(self.grid_size, self.reward_cell)

    def initialize_predictor(self):
        self.predictor = predictors.TrivialPredictor(self.system._compute_dynamics)

    def initialize_optimizers(self):
        self.actor_optimizer = optimizers.BruteForceOptimizer(5, action_space)

    def initialize_models(self):
        self.actor_model = models.LookupTable(self.grid_size)
        self.critic_model = models.LookupTable(self.grid_size)
        self.running_objective_model = models.LookupTable(
            self.grid_size, action_space_size
        )

        self.actor_model.weights = rc.ones(self.grid_size)

    def initialize_objectives(self):
        self.running_objective = objectives.RunningObjective(
            self.running_objective_model
        )

        self.running_objective.model.weights = -1 * rc.ones(
            (self.grid_size + [action_space_size])
        )

        self.starting_cell_xy = [1, 1]
        self.running_objective.model.weights[
            self.reward_cell[0], self.reward_cell[1], :
        ] = 100
        self.running_objective.model.weights[5, 5:8, :] = punishment
        self.running_objective.model.weights[1:6, 7, :] = punishment
        self.running_objective.model.weights[1, 3:8, :] = punishment
        self.running_objective.model.weights[1:6, 3, :] = punishment
        self.running_objective.model.weights[3:6, 5, :] = punishment
        self.running_objective.model.weights[6, 5, :] = punishment
        self.running_objective.model.weights[2, 1:3, :] = punishment
        self.running_objective.model.weights[4, 0:2, :] = punishment
        self.running_objective.model.weights[6, 1:3, :] = punishment

        self.punishment_cells = []
        self.punishment_cells.extend(slice2indices(slice(5, 8, 1), fixed_x=5))
        self.punishment_cells.extend(slice2indices(slice(1, 6, 1), fixed_y=7))
        self.punishment_cells.extend(slice2indices(slice(3, 8, 1), fixed_x=1))
        self.punishment_cells.extend(slice2indices(slice(1, 6, 1), fixed_y=3))
        self.punishment_cells.extend(slice2indices(slice(3, 6, 1), fixed_y=5))
        self.punishment_cells.extend(slice2indices(slice(1, 3, 1), fixed_x=2))
        self.punishment_cells.extend(slice2indices(slice(0, 2, 1), fixed_x=4))
        self.punishment_cells.extend(slice2indices(slice(1, 3, 1), fixed_x=6))
        self.punishment_cells.extend([[6, 5]])

    def initialize_actor_critic(self):
        self.critic = CriticTabularVI(
            dim_state_space=self.grid_size,
            running_objective=self.running_objective,
            predictor=self.predictor,
            model=self.critic_model,
            actor_model=self.actor_model,
            discount_factor=self.discount_factor,
            terminal_state=self.reward_cell,
        )
        self.actor = ActorTabular(
            dim_world=self.grid_size,
            predictor=self.predictor,
            optimizer=self.actor_optimizer,
            running_objective=self.running_objective,
            model=self.actor_model,
            action_space=action_space,
            critic=self.critic,
            discount_factor=self.discount_factor,
            terminal_state=self.reward_cell,
        )

    def initialize_scenario(self):
        self.scenario = TabularScenarioVI(self.actor, self.critic, 400)

    def initialize_visualizer(self):
        self.visualizer = animators.AnimatorGridWorld(
            self.actor,
            self.critic,
            self.reward_cell,
            self.starting_cell_xy,
            self.punishment_cells,
            self.scenario,
        )

    def execute_pipeline(self, **kwargs):
        self.load_config()
        self.setup_env()
        self.__dict__.update(kwargs)
        self.initialize_system()
        self.initialize_predictor()
        self.initialize_optimizers()
        self.initialize_models()
        self.initialize_objectives()
        self.initialize_actor_critic()
        self.initialize_scenario()
        self.initialize_visualizer()

        if self.no_visual:
            self.scenario.run()
        else:
            anm = animation.FuncAnimation(
                self.visualizer.fig_sim,
                self.visualizer.animate,
                blit=False,
                repeat=False,
                interval=0.001,
            )

            self.visualizer.get_anm(anm)

            cId = self.visualizer.fig_sim.canvas.mpl_connect(
                "key_press_event", lambda event: on_key_press(event, anm)
            )

            anm.running = True

            self.visualizer.fig_sim.tight_layout()

            plt.show()


if __name__ == "__main__":

    PipelineTabular().execute_pipeline()
