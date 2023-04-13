try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    from unittest.mock import MagicMock

    torch = MagicMock()

import numpy as np


class EpisodeBuffer:
    def __init__(self):
        self.nullify_buffer()

    def nullify_buffer(self):
        self.episode_ids = []
        self.observations = []
        self.actions = []
        self.running_objectives = []
        self.current_total_objectives = []
        self.total_objectives = np.array([])
        self.is_step_dones = []
        self.objectives_acc_stats = None

    def add_step_data(
        self,
        observation,
        action,
        running_objective,
        current_total_objective,
        episode_id,
        is_step_done,
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.running_objectives.append(running_objective)
        self.current_total_objectives.append(current_total_objective)
        self.episode_ids.append(int(episode_id))
        self.is_step_dones.append(is_step_done)

    def add_total_objective(self, total_objective):
        self.total_objectives = np.append(self.total_objectives, total_objective)


class ObservationActionObjectiveAccStatsDataset(Dataset, EpisodeBuffer):
    def __init__(
        self,
        system,
        is_use_derivative,
        is_cat_action,
        is_tail_sum_running_objectives=False,
    ) -> None:
        super(ObservationActionObjectiveAccStatsDataset, self).__init__()
        self.system = system
        self.derivative = system.compute_dynamics
        self.is_use_derivative = is_use_derivative
        self.is_cat_action = is_cat_action
        self.is_tail_sum_running_objectives = is_tail_sum_running_objectives

    def __len__(self):
        return len(self.observations)

    def calculate_objective_acc_stats(self):
        episode_ids = np.array(self.episode_ids)
        if self.is_tail_sum_running_objectives:
            running_objectives = np.array(self.running_objectives)
            current_total_objectives = np.array(self.current_total_objectives)
            return (
                self.total_objectives[episode_ids - 1]
                - current_total_objectives
                + running_objectives
            )
        else:
            return self.total_objectives[episode_ids - 1]

    def __getitem__(self, idx):
        if self.objectives_acc_stats is None:
            self.objectives_acc_stats = self.calculate_objective_acc_stats()

        observation_for_actor, observation_for_critic = (
            torch.tensor(self.observations[idx]),
            torch.tensor(self.observations[idx]),
        )
        action = torch.tensor(self.actions[idx])

        if self.is_use_derivative:
            derivative = self.derivative([], observation_for_actor, action)
            observation_for_actor = torch.cat([self.observations[idx], derivative])

        if self.is_cat_action:
            return {
                "observations_actions_for_actor": torch.cat(
                    [observation_for_actor, action]
                ),
                "observations_actions_for_critic": torch.cat(
                    [observation_for_critic, action]
                ),
                "objective_acc_stats": torch.tensor(self.objectives_acc_stats[idx]),
            }
        else:
            return {
                "observations_for_actor": observation_for_actor,
                "observations_for_critic": observation_for_critic,
                "objective_acc_stats": torch.tensor(self.objectives_acc_stats[idx]),
            }
