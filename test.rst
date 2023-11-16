ReinforcePipeline is an implementation of the REINFORCE algorithm, a Monte Carlo policy gradient method for reinforcement learning.

This pipeline updates policies via gradient descent on expected costs. The implementation can include a baseline to reduce variance and can
be configured to avoid letting past decisions affect current updates.

Below we provide the full listing of the algorithm:

General formula:

.. math::
    \theta_{i+1} \leftarrow \theta_{i} - \alpha_i \mathbb{E}\left[  \sum_{k = 0}^{N-1} \left( \sum_{k'=k}^{N-1} \gamma^{k'}r(Y_{k'}, U_{k'}) - B_k \right) 
    \nabla_{\theta} \ln\rho^{\theta}(U_k \mid Y_k)\big|_{\theta=\theta^i}\right].

where :math:`B_k` is the so-called baseline --- a random variable independent on :math:`(Y_k, U_k)` (or in general case independent on 
:math:`U_k` conditioned on :math:`Y_k`).

- [Optional] initialize baseline for the 1st iteration: :math:`b^1 = 0.`

- for :math:`i` in :math:`\{1, \dots, \mathcal I\}`: (:math:`\mathcal I` is the number of iterations)
    - for :math:`j` in :math:`\{1, \dots, M\}`: (:math:`j` is the number of episode)
        - for :math:`k` in :math:`\{0, \dots, N-1\}`: (:math:`k` is the number of step inside an episode)
            - obtain observation :math:`y_k^j` from the system
            - sample action :math:`u_k^j` from :math:`\rho^{\theta}(u_k^j \mid y_k^j)`

    - [optional] compute a baseline for the next iteration, e.g., as previous means of tail objectives (a common choice):

    .. math::

        b^{i + 1}_k \leftarrow \frac{1}{M} \sum_{j=1}^M \sum_{k'=k}^{N-1} \gamma^{k'} r(y_{k'}^j, u_{k'}^j)

    (it is our approach you can put any baseline you want)

    - perform a gradient step:

    .. math::

        \theta_{i+1} \leftarrow \theta_i - \alpha_i \frac{1}{M} \sum_{j=1}^{M} \sum_{k=0}^{N-1} \left(\sum_{k'=k}^{N-1} \gamma^{k'} 
        r\left(y_{k'}^j, u_{k'}^j\right) - b^i_k\right)\nabla_\theta \ln\rho^\theta(u_k^j \mid y_k^j)\big|_{\theta=\theta^i}

**Usage Example**

.. code-block:: python

    import regelum as rg
    import numpy as np
    import torch
    from regelum.system import InvertedPendulumPD
    from regelum.objective import RunningObjective
    from regelum.simulator import CasADi
    from regelum.pipeline import ReinforcePipeline

    # Initialize the system and running objective
    system = InvertedPendulumPD()
    running_objective = RunningObjective(
        model=rg.model.ModelQuadLin("diagonal", weights=[10, 3.0, 0.0])
    )

    # Instantiate the ReinforcePipeline
    pipeline = ReinforcePipeline(
        policy_model=rg.model.PerceptronWithTruncatedNormalNoise(
            dim_input=system.dim_observation,
            dim_hidden=4,
            n_hidden_layers=2,
            dim_output=system.dim_inputs,
            hidden_activation=torch.nn.LeakyReLU(0.2),
            output_activation=rg.model.MultiplyByConstant(1 / 100),
            output_bounds=system.action_bounds,
            stds=0.1
            * (
                np.array(system.action_bounds)[None, :, 1]
                - np.array(system.action_bounds)[None, :, 0]
            )
            / 2.0,
            is_truncated_to_output_bounds=True,
        ),
        simulator=CasADi(
            system=system,
            state_init=np.array([[3.14, 0]]),
            time_final=10,
            max_step=0.001,
        ),
        policy_opt_method_kwargs={"lr": 0.1},
        sampling_time=0.01,
        running_objective=running_objective,
        N_episodes=4,
        N_iterations=100,
        is_with_baseline=True,
        is_do_not_let_the_past_distract_you=True,
    )

    # Run the training process
    pipeline.run()
    