import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)


import rcognita as r
from omegaconf import DictConfig, OmegaConf, flag_override
from rcognita.visualization.vis_3wrobot import Animator3WRobot
from rcognita.visualization.vis_grid_world import AnimatorGridWorld
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

reward_cell = [4, 6]
punishment = -1e2
grid_size = [9, 9]


def slice2indices(sl, fixed_x=None, fixed_y=None):
    min_ind = sl.start
    max_ind = sl.stop
    if fixed_x:
        return [[fixed_x, sl.indices(i)[1]] for i in range(min_ind, max_ind)]
    elif fixed_y:
        return [[sl.indices(i)[1], fixed_y] for i in range(min_ind, max_ind)]


@r.main(config_name="scenario",)
def launch(scenario_config):
    scenario = ~scenario_config
    scenario.actor.running_objective.model.weights = -1 * np.ones(
        (scenario.actor.dim_world + [len(scenario.actor.action_space)])
    )
    scenario.actor.running_objective.model.weights[
        reward_cell[0], reward_cell[1], :
    ] = 100
    scenario.actor.running_objective.model.weights[5, 5:8, :] = punishment
    scenario.actor.running_objective.model.weights[1:6, 7, :] = punishment
    scenario.actor.running_objective.model.weights[1, 3:8, :] = punishment
    scenario.actor.running_objective.model.weights[1:6, 3, :] = punishment
    scenario.actor.running_objective.model.weights[3:6, 5, :] = punishment
    scenario.actor.running_objective.model.weights[6, 5, :] = punishment
    scenario.actor.running_objective.model.weights[2, 1:3, :] = punishment
    scenario.actor.running_objective.model.weights[4, 0:2, :] = punishment
    scenario.actor.running_objective.model.weights[6, 1:3, :] = punishment

    scenario.critic.running_objective.model.weights[5, 5:8, :] = punishment
    scenario.critic.running_objective.model.weights[1:6, 7, :] = punishment
    scenario.critic.running_objective.model.weights[1, 3:8, :] = punishment
    scenario.critic.running_objective.model.weights[1:6, 3, :] = punishment
    scenario.critic.running_objective.model.weights[3:6, 5, :] = punishment
    scenario.critic.running_objective.model.weights[6, 5, :] = punishment
    scenario.critic.running_objective.model.weights[2, 1:3, :] = punishment
    scenario.critic.running_objective.model.weights[4, 0:2, :] = punishment
    scenario.critic.running_objective.model.weights[6, 1:3, :] = punishment

    punishment_cells = []
    punishment_cells.extend(slice2indices(slice(5, 8, 1), fixed_x=5))
    punishment_cells.extend(slice2indices(slice(1, 6, 1), fixed_y=7))
    punishment_cells.extend(slice2indices(slice(3, 8, 1), fixed_x=1))
    punishment_cells.extend(slice2indices(slice(1, 6, 1), fixed_y=3))
    punishment_cells.extend(slice2indices(slice(3, 6, 1), fixed_y=5))
    punishment_cells.extend(slice2indices(slice(1, 3, 1), fixed_x=2))
    punishment_cells.extend(slice2indices(slice(0, 2, 1), fixed_x=4))
    punishment_cells.extend(slice2indices(slice(1, 3, 1), fixed_x=6))
    punishment_cells.extend([[6, 5]])

    animator = AnimatorGridWorld(
        scenario=scenario,
        reward_cell_xy=scenario.actor.terminal_state,
        starting_cell_xy=[1, 1],
        punishment_cells=punishment_cells,
    )
    anm = animation.FuncAnimation(
        animator.fig_sim, animator.animate, blit=False, repeat=False, interval=0.001,
    )

    animator.get_anm(anm)

    anm.running = True

    animator.fig_sim.tight_layout()

    plt.show()


if __name__ == "__main__":
    launch()
