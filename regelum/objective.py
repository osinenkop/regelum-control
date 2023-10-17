"""Module that contains general objectives functions that can be used by various entities of the framework.

For instance, a running objective can be used commonly by a generic optimal controller, an actor, a critic, a logger, an animator, a pipeline etc.

"""

from abc import ABC, abstractmethod

import regelum
from .model import Model, PerceptronWithNormalNoise, ModelNN
from typing import Optional, Union
import torch
from .__utilities import rc
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

        :param model: function that calculates the running objective for a given observation and action.
        :type model: function
        """
        self.model = (lambda observation, action: 0) if model is None else model

    def __call__(self, observation, action, is_save_batch_format=False):
        """Calculate the running objective for a given observation and action.

        :param observation: current observation.
        :type observation: numpy array
        :param action: current action.
        :type action: numpy array
        :return: running objective value.
        :rtype: float
        """
        running_objective = self.model(observation, action)
        if not is_save_batch_format:
            return running_objective[0][0]

        return running_objective


def reinforce_objective(
    policy_model: PerceptronWithNormalNoise,
    observations: torch.FloatTensor,
    actions: torch.FloatTensor,
    tail_total_objectives: torch.FloatTensor,
    total_objectives: torch.FloatTensor,
    baselines: torch.FloatTensor,
    is_with_baseline: bool,
    is_do_not_let_the_past_distract_you: bool,
    device: Union[str, torch.device],
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


    :param policy_model: The policy model used to calculate the log probabilities.
    :type policy_model: GaussianPDFModel
    :param observations: The observations tensor.
    :type observations: torch.FloatTensor
    :param actions: The actions tensor.
    :type actions: torch.FloatTensor
    :param tail_total_objectives: The tail total objectives tensor.
    :type tail_total_objectives: torch.FloatTensor
    :param total_objectives: The total objectives tensor.
    :type total_objectives: torch.FloatTensor
    :param baselines: The baselines tensor.
    :type baselines: torch.FloatTensor
    :param is_with_baseline: Flag indicating whether to subtract baselines from the target objectives.
    :type is_with_baseline: bool
    :param is_do_not_let_the_past_distract_you: Flag indicating whether to use tail total objectives.
    :type is_do_not_let_the_past_distract_you: bool
    :param device: The device to use for the calculations.
    :type device: Union[str, torch.device]
    :param N_episodes: The number of episodes.
    :type N_episodes: int

    :return: surrogate objective value.
    :rtype: torch.FloatTensor
    """
    log_pdfs = policy_model.log_pdf(observations.to(device), actions.to(device))
    if is_do_not_let_the_past_distract_you:
        target_objectives = tail_total_objectives.to(device)
    else:
        target_objectives = total_objectives.to(device)
    if is_with_baseline:
        target_objectives -= baselines.to(device)

    return (log_pdfs * target_objectives).sum() / N_episodes


def sdpg_objective(
    policy_model: PerceptronWithNormalNoise,
    critic_model: ModelNN,
    observations: torch.FloatTensor,
    actions: torch.FloatTensor,
    timestamps: torch.FloatTensor,
    episode_ids: torch.LongTensor,
    device: Union[str, torch.device],
    discount_factor: float,
    N_episodes: int,
    running_objectives: torch.FloatTensor,
) -> torch.FloatTensor:
    """Calculate the sum of the objective function values for the Stochastic Deterministic Policy Gradient (SDPG) algorithm.

    TODO: add link to papers + latex code for objective function

    :param policy_model: The policy model that represents the probability density function (PDF) of the action given the observation.
    :type policy_model: PerceptronWithNormalNoise
    :param critic_model: The critic model that estimates the Q-function for a given observation-action pair.
    :type critic_model: ModelNN
    :param observations: The tensor containing the observations.
    :type observations: torch.FloatTensor
    :param actions: The tensor containing the actions.
    :type actions: torch.FloatTensor
    :param timestamps: The tensor containing the timestamps.
    :type timestamps: torch.FloatTensor
    :param episode_ids: Episode ids.
    :type episode_ids: torch.LongTensor
    :param device: The device on which the computations are performed.
    :type device: Union[str, torch.device]
    :param discount_factor: The discount factor used to discount future running objectives.
    :type discount_factor: float
    :param N_episodes: The number of episodes used to estimate the expected objective function value.
    :type N_episodes: int

    :return: SDPG surrogate objective.
    :rtype: torch.FloatTensor
    """
    critic_values = critic_model(observations)
    log_pdfs = policy_model.log_pdf(observations.to(device), actions.to(device))

    objective = 0.0
    for episode_idx in torch.unique(episode_ids):
        mask = episode_ids == episode_idx
        advantages = (
            running_objectives[mask][:-1]
            + discount_factor * critic_values[mask][1:]
            - critic_values[mask][:-1]
        )

        objective += (
            discount_factor ** timestamps[mask][:-1]
            * advantages
            * log_pdfs[mask.reshape(-1)][:-1]
        ).sum()

    return objective / N_episodes


def ppo_objective(
    policy_model: PerceptronWithNormalNoise,
    critic_model: ModelNN,
    observations: torch.FloatTensor,
    actions: torch.FloatTensor,
    timestamps: torch.FloatTensor,
    episode_ids: torch.LongTensor,
    device: Union[str, torch.device],
    discount_factor: float,
    N_episodes: int,
    running_objectives: torch.FloatTensor,
    epsilon: float,
    initial_log_probs: torch.FloatTensor,
    running_objective_type: str,
) -> torch.FloatTensor:
    """Calculate PPO objective.

    TODO: Write docsting with for ppo objective.

    :param policy_model: policy model
    :type policy_model: PerceptronWithNormalNoise
    :param critic_model: critic model
    :type critic_model: ModelNN
    :param observations: tensor of observations in iteration
    :type observations: torch.FloatTensor
    :param actions: tensor of actions in iteration
    :type actions: torch.FloatTensor
    :param timestamps: timestamps
    :type timestamps: torch.FloatTensor
    :param episode_ids: episode ids
    :type episode_ids: torch.LongTensor
    :param device: device (cuda or cpu)
    :type device: Union[str, torch.device]
    :param discount_factor: discount factor
    :type discount_factor: float
    :param N_episodes: number of episodes
    :type N_episodes: int
    :param running_objectives: tensor of running objectives (either costs or rewards)
    :type running_objectives: torch.FloatTensor
    :param epsilon: epsilon
    :type epsilon: float
    :param initial_log_probs:
    :type initial_log_probs: torch.FloatTensor
    :param running_objective_type: can be either `cost` or `reward`
    :type running_objective_type: str
    :return: objective for PPO
    :rtype: torch.FloatTensor
    """
    assert (
        running_objective_type == "cost" or running_objective_type == "reward"
    ), "running_objective_type can be either 'cost' or 'reward'"

    critic_values = critic_model(observations)
    prob_ratios = torch.exp(
        policy_model.log_pdf(observations.to(device), actions.to(device))
        - initial_log_probs.reshape(-1)
    ).reshape(-1, 1)
    clipped_prob_ratios = torch.clamp(prob_ratios, 1 - epsilon, 1 + epsilon)
    objective_value = 0.0
    for episode_idx in torch.unique(episode_ids):
        mask = episode_ids.reshape(-1) == episode_idx
        advantages = (
            running_objectives[mask][:-1]
            + discount_factor * critic_values[mask][1:]
            - critic_values[mask][:-1]
        )

        objective_value += (
            torch.sum(
                (discount_factor ** timestamps[mask][:-1])
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
    device: Union[str, torch.device],
) -> torch.FloatTensor:
    """Calculate the objective value for the DDPG algorithm.

    TODO: add link to papers + latex code for objective function

    :param policy_model: The policy model that generates actions based on observations.
    :type policy_model: GaussianPDFModel
    :param critic_model: The critic model that approximates the Q-function.
    :type critic_model: ModelNN
    :param observations: The batch of observations.
    :type observations: torch.FloatTensor
    :param device: The device to perform computations on.
    :type device: Union[str, torch.device]

    :return: The objective value.
    :rtype: torch.FloatTensor
    """
    observations_on_device = observations.to(device)
    return critic_model(
        torch.cat(
            [observations_on_device, policy_model(observations_on_device)],
            dim=1,
        )
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

    :param critic_model: Q function model.
    :type critic_model: ModelNN
    :param observation: Batch of observations.
    :type observation: torch.FloatTensor
    :param action: Batch of actions.
    :type action: torch.FloatTensor
    :param running_objective: Batch of running objectives.
    :type running_objective: torch.FloatTensor
    :param td_n: Number of temporal difference steps.
    :type td_n: int
    :param discount_factor: Discount factor for future running objectives.
    :type discount_factor: float
    :param device: Device to proceed computations on.
    :type device: Union[str, torch.device]
    :param sampling_time: Sampling time for discounting.
    :type sampling_time: float
    :param is_use_same_critic: Whether to use the critic model from the previous iteration or not (`True` for not using, `False` for using).
    :type is_use_same_critic: bool
    :return: objective value
    :rtype: torch.FloatTensor
    """
    batch_size = running_objective.shape[0]
    assert batch_size > td_n, f"batch size {batch_size} too small for such td_n {td_n}"
    discount_factors = rc.array(
        [[discount_factor ** (sampling_time * i)] for i in range(td_n)],
        prototype=running_objective,
        _force_numeric=True,
    )
    discounted_tdn_sum_of_running_objectives = rc.vstack(
        [
            rc.sum(running_objective[i : i + td_n, :] * discount_factors)
            for i in range(batch_size - td_n)
        ]
    )

    if critic_targets is None:
        critic_targets = critic_model_output[td_n:, :]
    else:
        critic_targets = critic_targets[td_n:, :]

    temporal_difference = rc.mean(
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
    policy_model_weights,
    predictor: Predictor,
    running_objective,
    model,
    prediction_horizon,
    discount_factor=1.0,
):
    (
        observation_sequence_predicted,
        action_sequence_predicted,
    ) = predictor.predict_state_sequence_from_model(
        observation,
        prediction_horizon=prediction_horizon,
        model=model,
        model_weights=policy_model_weights,
    )

    observation_sequence = rc.vstack(
        (rc.force_row(observation), observation_sequence_predicted[:-1, :])
    )

    running_objectives = running_objective(
        observation_sequence, action_sequence_predicted, is_save_batch_format=True
    )

    if discount_factor == 1.0:
        discount_factors = rc.array(
            [
                [discount_factor ** (predictor.pred_step_size * i)]
                for i in range(prediction_horizon)
            ],
            prototype=observation,
            _force_numeric=True,
        )
        mpc_objective_value = rc.sum(running_objectives * discount_factors)
    else:
        mpc_objective_value = rc.sum(running_objectives)

    return mpc_objective_value


def rpo_objective(
    observation,
    policy_model_weights,
    predictor: Predictor,
    running_objective,
    model,
    prediction_horizon,
    discount_factor,
    critic,
    critic_weights,
):
    rpo_objective_value = 0
    (
        observation_sequence_predicted,
        action_sequence_predicted,
    ) = predictor.predict_state_sequence_from_model(
        observation,
        prediction_horizon=prediction_horizon,
        model=model,
        model_weights=policy_model_weights,
    )
    observation_sequence = rc.vstack(
        (rc.force_row(observation), observation_sequence_predicted[:-1, :])
    )

    discount_factors = rc.array(
        [
            [discount_factor ** (predictor.pred_step_size * i)]
            for i in range(prediction_horizon)
        ],
        prototype=observation,
        _force_numeric=True,
    )

    rpo_objective_value = rc.sum(
        discount_factors
        * running_objective(
            observation_sequence, action_sequence_predicted, is_save_batch_format=True
        )
    )

    observation_last = observation_sequence_predicted[-1, :]
    rpo_objective_value += rc.sum(
        discount_factor ** (predictor.pred_step_size * prediction_horizon)
        * critic(observation_last, weights=critic_weights)
    )

    return rpo_objective_value


def rql_objective(
    observation,
    policy_model_weights,
    predictor: Predictor,
    running_objective,
    model,
    prediction_horizon,
    discount_factor,
    critic,
    critic_weights,
):
    assert prediction_horizon >= 2, "rql only works for prediction_horizon >= 2"
    rql_objective_value = 0
    (
        observation_sequence_predicted,
        action_sequence_predicted,
    ) = predictor.predict_state_sequence_from_model(
        observation,
        prediction_horizon=prediction_horizon,
        model=model,
        model_weights=policy_model_weights,
    )
    observation_sequence = rc.vstack(
        (rc.force_row(observation), observation_sequence_predicted[:-2, :])
    )
    discount_factors = rc.array(
        [
            [discount_factor ** (predictor.pred_step_size * i)]
            for i in range(prediction_horizon - 1)
        ],
        prototype=observation,
        _force_numeric=True,
    )
    rql_objective_value = rc.sum(
        discount_factors
        * running_objective(
            observation_sequence,
            action_sequence_predicted[:-1, :],
            is_save_batch_format=True,
        )
    )

    observation_last = observation_sequence_predicted[-2, :]
    rql_objective_value += rc.sum(
        discount_factor ** (predictor.pred_step_size * (prediction_horizon - 1))
        * critic(
            observation_last, action_sequence_predicted[-1, :], weights=critic_weights
        )
    )

    return rql_objective_value


def sql_objective(
    observation,
    policy_model_weights,
    predictor: Predictor,
    model,
    prediction_horizon,
    critic,
    critic_weights=None,
):
    (
        observation_sequence_predicted,
        action_sequence_predicted,
    ) = predictor.predict_state_sequence_from_model(
        observation,
        prediction_horizon=prediction_horizon,
        model=model,
        model_weights=policy_model_weights,
    )

    observation_sequence = rc.vstack(
        (rc.force_row(observation), observation_sequence_predicted[:-1, :])
    )

    sql_objective_value = rc.sum(
        rc.sum(
            critic(
                observation_sequence, action_sequence_predicted, weights=critic_weights
            )
        )
    )

    return sql_objective_value
