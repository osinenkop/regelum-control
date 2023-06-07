try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Sampler
except ImportError:
    from unittest.mock import MagicMock

    torch = MagicMock()

import numpy as np
import pandas as pd
import random
from abc import ABC, abstractmethod


# class EpisodeBuffer:
#     def __init__(self):
#         self.nullify_buffer()

#     def nullify_buffer(self):
#         self.episode_ids = []
#         self.observations = []
#         self.actions = []
#         self.running_objectives = []
#         self.current_total_objectives = []
#         self.total_objectives = np.array([])
#         self.is_step_dones = []
#         self.objectives_acc_stats = None

#     def add_step_data(
#         self,
#         observation,
#         action,
#         running_objective,
#         current_total_objective,
#         episode_id,
#         is_step_done,
#     ):
#         self.observations.append(observation)
#         self.actions.append(action)
#         self.running_objectives.append(running_objective)
#         self.current_total_objectives.append(current_total_objective)
#         self.episode_ids.append(int(episode_id))
#         self.is_step_dones.append(is_step_done)

#         self.episodes_lengths = None

#     def set_total_objectives_of_episodes(self, total_objectives):
#         self.total_objectives = np.array(total_objectives)

#     def get_episodes_lengths(self):
#         if self.episodes_lengths is None:
#             values, counts = np.unique(self.episode_ids, return_counts=True)
#             self.episodes_lengths = pd.Series(index=values, data=counts)

#         return self.episodes_lengths

#     def transform_to_raw_idx(self, step_idx, episode_id):
#         if len(self.get_episodes_lengths()) == 1:
#             return step_idx
#         else:
#             return int(
#                 self.get_episodes_lengths().loc[: episode_id - 1].sum() + step_idx
#             )


# class ObservationActionObjectiveAccStatsDataset(Dataset, EpisodeBuffer):
#     def __init__(
#         self,
#         system,
#         is_use_derivative,
#         is_cat_action,
#         is_tail_sum_running_objectives=False,
#     ) -> None:
#         super(ObservationActionObjectiveAccStatsDataset, self).__init__()
#         self.system = system
#         self.derivative = system.compute_dynamics
#         self.is_use_derivative = is_use_derivative
#         self.is_cat_action = is_cat_action
#         self.is_tail_sum_running_objectives = is_tail_sum_running_objectives

#         self.objectives_acc_stats = None

#     def __len__(self):
#         return len(self.observations)

#     def calculate_objective_acc_stats(self):
#         episode_ids = np.array(self.episode_ids)
#         if self.is_tail_sum_running_objectives:
#             running_objectives = np.array(self.running_objectives)
#             current_total_objectives = np.array(self.current_total_objectives)
#             return (
#                 self.total_objectives[episode_ids]
#                 - current_total_objectives
#                 + running_objectives
#             )
#         else:
#             return self.total_objectives[episode_ids]

#     def __getitem__(self, idx):
#         if self.objectives_acc_stats is None:
#             self.objectives_acc_stats = self.calculate_objective_acc_stats()

#         observation_for_actor, observation_for_critic = (
#             torch.tensor(self.observations[idx]),
#             torch.tensor(self.observations[idx]),
#         )
#         action = torch.tensor(self.actions[idx])

#         if self.is_use_derivative:
#             derivative = self.derivative([], observation_for_actor, action)
#             observation_for_actor = torch.cat([observation_for_actor, derivative])

#         if self.is_cat_action:
#             return {
#                 "observations_actions_for_actor": torch.cat(
#                     [observation_for_actor, action]
#                 ),
#                 "observations_actions_for_critic": torch.cat(
#                     [observation_for_critic, action]
#                 ),
#                 "objective_acc_stats": torch.tensor(self.objectives_acc_stats[idx]),
#             }
#         else:
#             return {
#                 "observations_for_actor": observation_for_actor,
#                 "observations_for_critic": observation_for_critic,
#                 "objective_acc_stats": torch.tensor(self.objectives_acc_stats[idx]),
#             }


class IterationBuffer(Dataset):
    """Buffer for experience replay."""

    def __init__(self) -> None:
        """Initialize `IterationBuffer`."""

        super().__init__()
        self.next_baseline = 0.0
        self.nullify_buffer()

    def nullify_buffer(self) -> None:
        """Clear all buffer data."""

        self.episode_ids = []
        self.observations = []
        self.actions = []
        self.running_objectives = []
        self.total_objectives = []
        self.timestamps = []
        self.last_total_objectives = None
        self.tail_total_objectives = None
        self.baseline = None

    def add_step_data(
        self,
        observation: np.array,
        timestamp: float,
        action: np.array,
        running_objective: float,
        current_total_objective: float,
        episode_id: int,
    ):
        """Add step data to experience replay

        Args:
            observation (np.array): current observation
            action (np.array): current action
            running_objective (float): current running objective
            current_total_objective (float): current total objective
            step_id (int): current step
            episode_id (int): current episode
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.timestamps.append(timestamp)
        self.running_objectives.append(running_objective)
        self.episode_ids.append(int(episode_id))
        self.total_objectives.append(current_total_objective)

    def get_N_episodes(self) -> int:
        """Get number of episodes

        Returns:
            int: number of episodes
        """
        return len(np.unique(self.episode_ids))

    def calculate_last_total_objectives(self):
        if self.last_total_objectives is None:
            self.last_total_objectives = (
                pd.Series(index=self.episode_ids, data=self.total_objectives)
                .groupby(level=0)
                .last()
                .loc[self.episode_ids]
                .values.reshape(-1)
            )

    def calculate_tail_total_objectives(
        self,
    ):
        """Calculate tail total costs and baseline.

        Returns:
            Tuple[np.array, float, float]: tuple of 3 elements tail_total_objectives, baseline, gradent_normalization_constant
        """

        if self.tail_total_objectives is None:
            groupby_episode_total_objectives = pd.Series(
                index=self.episode_ids, data=self.total_objectives
            ).groupby(level=0)

            self.tail_total_objectives = (
                groupby_episode_total_objectives.last()
                - groupby_episode_total_objectives.shift(periods=1, fill_value=0.0)
            ).values.reshape(-1)

    def calculate_baseline(self):
        if self.baseline is None:
            self.baseline = self.next_baseline
            self.next_baseline = np.mean(self.last_total_objectives)

    def __len__(self) -> int:
        """Get length of buffer. The method should be overrided due to inheritance from `torch.utils.data.Dataset`.

        Returns:
            int: length of buffer
        """
        return len(self.observations)

    def __getitem__(self, idx: int):
        """Get item with id `idx`. The method should be overrided due to inheritance from `torch.utils.data.Dataset`.

        Args:
            idx (int): id of dataset item to return

        Returns:
            Dict[str, torch.tensor]: dataset item, containing catted observation-action, tail total objective and baselines
        """

        self.calculate_last_total_objectives()
        self.calculate_tail_total_objectives()
        self.calculate_baseline()

        observation = torch.tensor(self.observations[idx], dtype=torch.float)
        action = torch.tensor(self.actions[idx], dtype=torch.float)

        return {
            "observations_actions": torch.cat([observation, action]),
            "total_objectives": torch.tensor(
                self.last_total_objectives[idx], dtype=torch.float
            ),
            "tail_total_objectives": torch.tensor(
                self.tail_total_objectives[idx], dtype=torch.float
            ),
            "baselines": torch.tensor(self.baseline, dtype=torch.float),
        }

    @property
    def data(self) -> pd.DataFrame:
        """Return current buffer content in pandas.DataFrame

        Returns:
            pd.DataFrame: current buffer content
        """

        return pd.DataFrame(
            {
                "episode_id": self.episode_ids,
                "time": self.timestamps,
                "observation": self.observations,
                "action": self.actions,
                "running_objective": self.running_objectives,
                "total_objective": self.total_objectives,
            }
        )
