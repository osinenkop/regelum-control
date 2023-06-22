# TODO: ALL DOCSTRINGS. WILL BE UNIFIED DATA HANDLER?

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

        self.episodes_lengths = None

    def set_total_objectives_of_episodes(self, total_objectives):
        self.total_objectives = np.array(total_objectives)

    def get_episodes_lengths(self):
        if self.episodes_lengths is None:
            values, counts = np.unique(self.episode_ids, return_counts=True)
            self.episodes_lengths = pd.Series(index=values, data=counts)

        return self.episodes_lengths

    def transform_to_raw_idx(self, step_idx, episode_id):
        if len(self.get_episodes_lengths()) == 1:
            return step_idx
        else:
            return int(
                self.get_episodes_lengths().loc[: episode_id - 1].sum() + step_idx
            )


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
        self.derivative = system.compute_state_dynamics
        self.is_use_derivative = is_use_derivative
        self.is_cat_action = is_cat_action
        self.is_tail_sum_running_objectives = is_tail_sum_running_objectives

        self.objectives_acc_stats = None

    def __len__(self):
        return len(self.observations)

    def calculate_objective_acc_stats(self):
        episode_ids = np.array(self.episode_ids)
        if self.is_tail_sum_running_objectives:
            running_objectives = np.array(self.running_objectives)
            current_total_objectives = np.array(self.current_total_objectives)
            return (
                self.total_objectives[episode_ids]
                - current_total_objectives
                + running_objectives
            )
        else:
            return self.total_objectives[episode_ids]

    def __getitem__(self, idx):
        if self.objectives_acc_stats is None:
            self.objectives_acc_stats = self.calculate_objective_acc_stats()

        observation_for_policy, observation_for_critic = (
            torch.tensor(self.observations[idx]),
            torch.tensor(self.observations[idx]),
        )
        action = torch.tensor(self.actions[idx])

        if self.is_use_derivative:
            derivative = self.derivative([], observation_for_policy, action)
            observation_for_policy = torch.cat([self.observations[idx], derivative])

        if self.is_cat_action:
            return {
                "observations_actions_for_policy": torch.cat(
                    [observation_for_policy, action]
                ),
                "observations_actions_for_critic": torch.cat(
                    [observation_for_critic, action]
                ),
                "objective_acc_stats": torch.tensor(self.objectives_acc_stats[idx]),
            }
        else:
            return {
                "observations_for_policy": observation_for_policy,
                "observations_for_critic": observation_for_critic,
                "objective_acc_stats": torch.tensor(self.objectives_acc_stats[idx]),
            }


class UpdatableSampler(Sampler, ABC):
    @abstractmethod
    def update_data(self, data_source, batch_size):
        pass


class EpisodicRandomSampler(UpdatableSampler):
    def __init__(
        self,
        n_batches=None,
    ) -> None:
        self.init_n_batches = n_batches
        self.episode_buffer_dataset = None

    def update_data(self, episode_buffer_dataset, batch_size):
        self.episode_buffer_dataset = episode_buffer_dataset
        self.batch_size = batch_size

        if self.init_n_batches is None:
            self.n_batches = int(np.ceil(len(episode_buffer_dataset) / self.batch_size))
        else:
            self.n_batches = int(self.init_n_batches)

    def __iter__(self):
        for _ in range(self.n_batches):
            episode_id = random.sample(
                self.episode_buffer_dataset.get_episodes_lengths().index.tolist(), k=1
            )[0]
            episode_len = self.episode_buffer_dataset.get_episodes_lengths().loc[
                episode_id
            ]

            yield iter(
                [
                    self.episode_buffer_dataset.transform_to_raw_idx(
                        step_idx=s,
                        episode_id=episode_id,
                    )
                    for s in np.random.randint(
                        low=0, high=episode_len - 1, size=self.batch_size
                    )
                ]
            )

    def __len__(self) -> int:
        return int(self.n_batches * self.batch_size)
