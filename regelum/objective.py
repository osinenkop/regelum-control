"""Module that contains general objectives functions that can be used by various entities of the framework.

For instance, a running objective can be used commonly by a generic optimal scenario, an actor, a critic, a logger, an animator, a scenario etc.

"""

from abc import ABC, abstractmethod

import regelum
from .model import Model, PerceptronWithTruncatedNormalNoise, ModelNN
from typing import Optional
import torch
from .utils import rg
from .predictor import Predictor


class Objective(regelum.RegelumBase, ABC):
    """A base class for objective implementations."""

    def __init__(self):
        """Initialize an instance of Objective."""
        pass

    @abstractmethod
    def __call__(self):
        pass


class RunningObjective(Objective):
    """Running reward/cost.

    In minimzations problems, it is called cost or loss, say.
    """

    def __init__(self, model: Optional[Model] = None):
        """Initialize a RunningObjective instance.

        Args:
        ----
            model (function): function that calculates the running
                objective for a given observation and action.
        """
        self.model = (lambda observation, action: 0) if model is None else model

    def __call__(self, observation, action, is_save_batch_format=False):
        """Calculate the running objective for a given observation and action.

        Args:
        ----
            observation (numpy array): current observation.
            action (numpy array): current action.

        Returns:
        -------
            float: running objective value.
        """
        running_objective = self.model(observation, action)
        if not is_save_batch_format:
            return running_objective[0][0]

        return running_objective


def reinforce_objective(
    policy_model: PerceptronWithTruncatedNormalNoise,
    observations: torch.FloatTensor,
    actions: torch.FloatTensor,
    tail_values: torch.FloatTensor,
    values: torch.FloatTensor,
    baselines: torch.FloatTensor,
    is_with_baseline: bool,
    is_do_not_let_the_past_distract_you: bool,
    N_episodes: int,
) -> torch.FloatTensor:
    r"""Calculate the surrogate objective for REINFORCE algorithm.

    Сalculates the following surrogate objective:
    :math:`\\frac{1}{M}\\sum_{j=1}^M \\sum_{k=0}^N \\log \\rho_{\\theta}(y_{k|j}, u_{k|j} \\left(\\sum_{k'=f_k}^N \\gamma^{k'} r(y_{k|j}, u_{k|j}) - B_k\\right),`
    where :math:`y_{k|j}` is the observation at time :math:`k` in episode :math:`j`, :math:`u_{k|j}` is the action at time :math:`k` in episode :math:`j`,
    :math:`\\rho_{\\theta}(u \\mid y)` is the probability density function of the policy model, :math:`f_k` is equal to :math:`k` if
    `is_do_not_let_the_past_distract_you` is `True`, and :math:`f_k` is equal to 0 if `is_do_not_let_the_past_distract_you` is `False`,
    :math: `B_k` is the baseline, which equals 0 if `is_with_baseline` is `False` and the total objective from previous iteration if `is_with_baseline` is `True`,
    :math: `M` is the number of episodes, :math:`N` is the number of actions.

    Args:
    ----
        policy_model (GaussianPDFModel): The policy model used to
            calculate the log probabilities.
        observations (torch.FloatTensor): The observations tensor.
        actions (torch.FloatTensor): The actions tensor.
        tail_values (torch.FloatTensor): The tail total objectives
            tensor.
        values (torch.FloatTensor): The total objectives tensor.
        baselines (torch.FloatTensor): The baselines tensor.
        is_with_baseline (bool): Flag indicating whether to subtract
            baselines from the target objectives.
        is_do_not_let_the_past_distract_you (bool): Flag indicating
            whether to use tail total objectives.
        device (Union[str, torch.device]): The device to use for the
            calculations.
        N_episodes (int): The number of episodes.

    Returns:
    -------
        torch.FloatTensor: surrogate objective value.
    """
    log_pdfs = policy_model.log_pdf(observations, actions)
    if is_do_not_let_the_past_distract_you:
        target_objectives = tail_values
    else:
        target_objectives = values
    if is_with_baseline:
        target_objectives -= baselines

    return (log_pdfs * target_objectives).sum() / N_episodes


def sdpg_objective(
    policy_model: PerceptronWithTruncatedNormalNoise,
    critic_model: ModelNN,
    observations: torch.FloatTensor,
    actions: torch.FloatTensor,
    times: torch.FloatTensor,
    episode_ids: torch.LongTensor,
    discount_factor: float,
    N_episodes: int,
    running_objectives: torch.FloatTensor,
    sampling_time: float,
) -> torch.FloatTensor:
    """Calculate the sum of the objective function values for the Stochastic Deterministic Policy Gradient (SDPG) algorithm.

    TODO: add link to papers + latex code for objective function

    Args:
    ----
        policy_model (PerceptronWithNormalNoise): The policy model that
            represents the probability density function (PDF) of the
            action given the observation.
        critic_model (ModelNN): The critic model that estimates the
            Q-function for a given observation-action pair.
        observations (torch.FloatTensor): The tensor containing the
            observations.
        actions (torch.FloatTensor): The tensor containing the actions.
        timestamps (torch.FloatTensor): The tensor containing the
            timestamps.
        episode_ids (torch.LongTensor): Episode ids.
        device (Union[str, torch.device]): The device on which the
            computations are performed.
        discount_factor (float): The discount factor used to discount
            future running objectives.
        N_episodes (int): The number of episodes used to estimate the
            expected objective function value.

    Returns:
    -------
        torch.FloatTensor: SDPG surrogate objective.
    """
    critic_values = critic_model(observations)
    log_pdfs = policy_model.log_pdf(observations, actions)

    objective = 0.0
    for episode_idx in torch.unique(episode_ids):
        mask = episode_ids == episode_idx
        advantages = (
            running_objectives[mask][:-1]
            + discount_factor**sampling_time * critic_values[mask][1:]
            - critic_values[mask][:-1]
        )

        objective += (
            discount_factor ** times[mask][:-1]
            * advantages
            * log_pdfs[mask.reshape(-1)][:-1]
        ).sum()

    return objective / N_episodes


def ppo_objective(
    policy_model: PerceptronWithTruncatedNormalNoise,
    critic_model: ModelNN,
    observations: torch.FloatTensor,
    actions: torch.FloatTensor,
    times: torch.FloatTensor,
    episode_ids: torch.LongTensor,
    discount_factor: float,
    N_episodes: int,
    running_objectives: torch.FloatTensor,
    cliprange: float,
    initial_log_probs: torch.FloatTensor,
    running_objective_type: str,
    sampling_time: float,
    gae_lambda: float,
) -> torch.FloatTensor:
    """Calculate PPO objective.

    TODO: Write docsting with for ppo objective.

    Args:
    ----
        policy_model (PerceptronWithNormalNoise): policy model
        critic_model (ModelNN): critic model
        observations (torch.FloatTensor): tensor of observations in
            iteration
        actions (torch.FloatTensor): tensor of actions in iteration
        timestamps (torch.FloatTensor): timestamps
        episode_ids (torch.LongTensor): episode ids
        device (Union[str, torch.device]): device (cuda or cpu)
        discount_factor (float): discount factor
        N_episodes (int): number of episodes
        running_objectives (torch.FloatTensor): tensor of running
            objectives (either costs or rewards)
        epsilon (float): epsilon
        initial_log_probs (torch.FloatTensor)
        running_objective_type (str): can be either `cost` or `reward`

    Returns:
    -------
        torch.FloatTensor: objective for PPO
    """
    assert (
        running_objective_type == "cost" or running_objective_type == "reward"
    ), "running_objective_type can be either 'cost' or 'reward'"

    critic_values = critic_model(observations)
    prob_ratios = torch.exp(
        policy_model.log_pdf(observations, actions) - initial_log_probs.reshape(-1)
    ).reshape(-1, 1)
    clipped_prob_ratios = torch.clamp(prob_ratios, 1 - cliprange, 1 + cliprange)
    objective_value = 0.0
    for episode_idx in torch.unique(episode_ids):
        mask = episode_ids.reshape(-1) == episode_idx
        deltas = (
            running_objectives[mask][:-1]
            + discount_factor**sampling_time * critic_values[mask][1:]
            - critic_values[mask][:-1]
        ).detach()

        if gae_lambda == 0.0:
            advantages = deltas
        else:
            gae_discount_factors = (gae_lambda * discount_factor) ** times[mask][:-1]
            reversed_gae_discounted_deltas = torch.flip(
                gae_discount_factors * deltas, dims=[0, 1]
            )
            advantages = (
                torch.flip(reversed_gae_discounted_deltas.cumsum(dim=0), dims=[0, 1])
                / gae_discount_factors
            )

        objective_value += (
            torch.sum(
                (discount_factor ** times[mask][:-1])
                * (
                    torch.maximum(
                        advantages * prob_ratios[mask][:-1],
                        advantages * clipped_prob_ratios[mask][:-1],
                    )
                    if running_objective_type == "cost"
                    else torch.minimum(
                        advantages * prob_ratios[mask][:-1],
                        advantages * clipped_prob_ratios[mask][:-1],
                    )
                )
            )
            / N_episodes
        )

    return objective_value


def ddpg_objective(
    policy_model: ModelNN,
    critic_model: ModelNN,
    observations: torch.FloatTensor,
) -> torch.FloatTensor:
    """Calculate the objective value for the DDPG algorithm.

    TODO: add link to papers + latex code for objective function

    Args:
    ----
        policy_model (GaussianPDFModel): The policy model that generates
            actions based on observations.
        critic_model (ModelNN): The critic model that approximates the
            Q-function.
        observations (torch.FloatTensor): The batch of observations.
        device (Union[str, torch.device]): The device to perform
            computations on.

    Returns:
    -------
        torch.FloatTensor: The objective value.
    """
    return critic_model(
        observations, policy_model.forward(observations, is_means_only=True)
    ).mean()


def temporal_difference_objective(
    critic_model_output: ModelNN,
    running_objective: torch.FloatTensor,
    td_n: int,
    discount_factor: float,
    sampling_time: float,
    critic_targets: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """Calculate temporal difference objective.

    Args:
    ----
        critic_model (ModelNN): Q function model.
        observation (torch.FloatTensor): Batch of observations.
        action (torch.FloatTensor): Batch of actions.
        running_objective (torch.FloatTensor): Batch of running
            objectives.
        td_n (int): Number of temporal difference steps.
        discount_factor (float): Discount factor for future running
            objectives.
        device (Union[str, torch.device]): Device to proceed
            computations on.
        sampling_time (float): Sampling time for discounting.
        is_use_same_critic (bool): Whether to use the critic model from
            the previous iteration or not (`True` for not using, `False`
            for using).

    Returns:
    -------
        torch.FloatTensor: objective value
    """
    batch_size = running_objective.shape[0]
    assert batch_size > td_n, f"batch size {batch_size} too small for such td_n {td_n}"
    discount_factors = rg.array(
        [[discount_factor ** (sampling_time * i)] for i in range(td_n)],
        prototype=running_objective,
        _force_numeric=True,
    )
    discounted_tdn_sum_of_running_objectives = rg.vstack(
        [
            rg.sum(running_objective[i : i + td_n, :] * discount_factors)
            for i in range(batch_size - td_n)
        ]
    )

    if critic_targets is None:
        critic_targets = critic_model_output[td_n:, :]
    else:
        critic_targets = critic_targets[td_n:, :]

    temporal_difference = rg.mean(
        (
            critic_model_output[:-td_n, :]
            - discounted_tdn_sum_of_running_objectives
            - discount_factor ** (td_n * sampling_time) * critic_targets
        )
        ** 2
    )
    return temporal_difference


def mpc_objective(
    observation,
    estimated_state,
    policy_model_weights,
    predictor: Predictor,
    running_objective,
    model,
    prediction_horizon,
    discount_factor=1.0,
):
    (
        state_sequence_predicted,
        action_sequence_predicted,
    ) = predictor.predict_state_sequence_from_model(
        estimated_state,
        prediction_horizon=prediction_horizon,
        model=model,
        model_weights=policy_model_weights,
        is_predict_last=False,
    )
    observation_sequence_predicted = predictor.system.get_observation(
        None,
        state=state_sequence_predicted,
        inputs=action_sequence_predicted,
        is_batch=True,
    )
    observation_sequence = rg.vstack(
        (rg.force_row(observation), observation_sequence_predicted)
    )

    running_objectives = running_objective(
        observation_sequence, action_sequence_predicted, is_save_batch_format=True
    )

    if discount_factor < 1.0:
        discount_factors = rg.array(
            [
                [discount_factor ** (predictor.pred_step_size * i)]
                for i in range(prediction_horizon + 1)
            ],
            prototype=observation,
            _force_numeric=True,
        )
        mpc_objective_value = rg.sum(running_objectives * discount_factors)
    else:
        mpc_objective_value = rg.sum(running_objectives)

    return mpc_objective_value


def rpv_objective(
    observation,
    estimated_state,
    policy_model_weights,
    predictor: Predictor,
    running_objective,
    model,
    prediction_horizon,
    discount_factor,
    critic,
    critic_weights,
):
    rpv_objective_value = 0
    (
        state_sequence_predicted,
        action_sequence_predicted,
    ) = predictor.predict_state_sequence_from_model(
        estimated_state,
        prediction_horizon=prediction_horizon,
        model=model,
        model_weights=policy_model_weights,
        is_predict_last=True,
    )
    observation_sequence_predicted = predictor.system.get_observation(
        None,
        state=state_sequence_predicted,
        inputs=action_sequence_predicted,
        is_batch=True,
    )
    observation_sequence = rg.vstack(
        (rg.force_row(observation), observation_sequence_predicted[:-1, :])
    )

    discount_factors = rg.array(
        [
            [discount_factor ** (predictor.pred_step_size * i)]
            for i in range(prediction_horizon)
        ],
        prototype=observation,
        _force_numeric=True,
    )

    rpv_objective_value = rg.sum(
        discount_factors
        * running_objective(
            observation_sequence, action_sequence_predicted, is_save_batch_format=True
        )
    )

    observation_last = observation_sequence_predicted[-1, :]
    rpv_objective_value += rg.sum(
        discount_factor ** (predictor.pred_step_size * prediction_horizon)
        * critic(observation_last, weights=critic_weights)
    )

    return rpv_objective_value


def rql_objective(
    observation,
    estimated_state,
    policy_model_weights,
    predictor: Predictor,
    running_objective,
    model,
    prediction_horizon,
    discount_factor,
    critic,
    critic_weights,
):
    rql_objective_value = 0
    (
        state_sequence_predicted,
        action_sequence_predicted,
    ) = predictor.predict_state_sequence_from_model(
        estimated_state,
        prediction_horizon=prediction_horizon,
        model=model,
        model_weights=policy_model_weights,
        is_predict_last=False,
    )

    if prediction_horizon == 0:
        return critic(observation, action_sequence_predicted, weights=critic_weights)
    observation_sequence_predicted = predictor.system.get_observation(
        None,
        state=state_sequence_predicted,
        inputs=action_sequence_predicted,
        is_batch=True,
    )
    observation_sequence = rg.vstack(
        (rg.force_row(observation), observation_sequence_predicted[:-1, :])
    )
    discount_factors = rg.array(
        [
            [discount_factor ** (predictor.pred_step_size * i)]
            for i in range(prediction_horizon)
        ],
        prototype=observation,
        _force_numeric=True,
    )
    rql_objective_value = rg.sum(
        discount_factors
        * running_objective(
            observation_sequence,
            action_sequence_predicted[:-1, :],
            is_save_batch_format=True,
        )
    )

    observation_last = state_sequence_predicted[-1, :]
    rql_objective_value += rg.sum(
        discount_factor ** (predictor.pred_step_size * prediction_horizon)
        * critic(
            observation_last, action_sequence_predicted[-1, :], weights=critic_weights
        )
    )

    return rql_objective_value


def sql_objective(
    observation,
    estimated_state,
    policy_model_weights,
    predictor: Predictor,
    model,
    prediction_horizon,
    critic,
    critic_weights=None,
):
    (
        state_sequence_predicted,
        action_sequence_predicted,
    ) = predictor.predict_state_sequence_from_model(
        estimated_state,
        prediction_horizon=prediction_horizon,
        model=model,
        model_weights=policy_model_weights,
        is_predict_last=False,
    )
    observation_sequence_predicted = predictor.system.get_observation(
        None,
        state=state_sequence_predicted,
        inputs=action_sequence_predicted,
        is_batch=True,
    )
    observation_sequence = rg.vstack(
        (rg.force_row(observation), observation_sequence_predicted)
    )

    sql_objective_value = rg.sum(
        rg.sum(
            critic(
                observation_sequence, action_sequence_predicted, weights=critic_weights
            )
        )
    )

    return sql_objective_value
