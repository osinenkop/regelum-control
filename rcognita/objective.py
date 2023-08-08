"""Module that contains general objectives functions that can be used by various entities of the framework.

For instance, a running objective can be used commonly by a generic optimal controller, an actor, a critic, a logger, an animator, a pipeline etc.

"""

from abc import ABC, abstractmethod

import rcognita
from .model import Model, PerceptronWithNormalNoise, ModelNN
from typing import Optional, Union
import torch


class Objective(rcognita.RcognitaBase, ABC):
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

    def __call__(self, observation, action):
        """Calculate the running objective for a given observation and action.

        :param observation: current observation.
        :type observation: numpy array
        :param action: current action.
        :type action: numpy array
        :return: running objective value.
        :rtype: float
        """
        running_objective = self.model(observation, action)

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
    device: Union[str, torch.device],
    discount_factor: float,
    N_episodes: int,
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
    :param device: The device on which the computations are performed.
    :type device: Union[str, torch.device]
    :param discount_factor: The discount factor used to discount future running objectives.
    :type discount_factor: float
    :param N_episodes: The number of episodes used to estimate the expected objective function value.
    :type N_episodes: int

    :return: SDPG surrogate objective.
    :rtype: torch.FloatTensor
    """
    observations_actions = torch.cat([observations, actions], dim=1).to(device)
    observations_zero_actions = torch.cat(
        [observations, torch.zeros_like(actions)],
        dim=1,
    ).to(device)

    with torch.no_grad():
        baseline = critic_model(observations_zero_actions)
        discounts = discount_factor ** timestamps.to(device)
        critic_value = discounts * (critic_model(observations_actions) - baseline)

    log_pdfs = policy_model.log_pdf(observations.to(device), actions.to(device))
    return (log_pdfs * critic_value).sum() / N_episodes


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
    critic_model: ModelNN,
    observation: torch.FloatTensor,
    running_objective: torch.FloatTensor,
    td_n: int,
    discount_factor: float,
    device: Union[str, torch.device],
    sampling_time: float,
    is_use_same_critic: bool,
    action: Optional[torch.FloatTensor] = None,
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
    batch_size = len(running_objective)
    assert batch_size > td_n, f"batch size {batch_size} too small for such td_n {td_n}"

    discount_factors = torch.FloatTensor(
        [discount_factor ** (sampling_time * i) for i in range(td_n)]
    )
    discounted_tdn_sum_of_running_objectives = torch.FloatTensor(
        [
            [(running_objective[i : i + td_n] * discount_factors).sum()]
            for i in range(batch_size - td_n)
        ]
    ).to(device)

    if action is not None:
        first_tdn_inputs = torch.cat([observation[:-td_n], action[:-td_n]], dim=1).to(
            device
        )
        last_tdn_inputs = torch.cat([observation[td_n:], action[td_n:]], dim=1).to(
            device
        )
    else:
        first_tdn_inputs = observation[:-td_n].to(device)
        last_tdn_inputs = observation[td_n:].to(device)

    if critic_targets is None:
        critic_targets = critic_model(
            last_tdn_inputs, use_stored_weights=not is_use_same_critic
        )
    else:
        critic_targets = critic_targets[-len(first_tdn_inputs) :].to(device)

    temporal_difference = (
        (
            critic_model(first_tdn_inputs)
            - discounted_tdn_sum_of_running_objectives
            - discount_factor ** (td_n * sampling_time) * critic_targets
        )
        ** 2
    ).mean()
    return temporal_difference
