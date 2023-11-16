REINFORCE Algorithm
===================

Overview
--------
The **ReinforcePipeline** implements the **REINFORCE** algorithm, which is a Monte Carlo policy gradient method used in reinforcement learning. This method seeks to optimize the policy by updating parameters in the direction that maximizes expected rewards (or minimizes the expected costs). 

.. note::
    We provide functionality for policy actions to be generated from a truncated normal distribution that is truncated to the action bounds provided in the system as default parameters.

Gradient Update Formulas
------------------------

Let us denote :math:`\theta_i` as policy weights at :math:`i`-th iteration, :math:`B_k` as the baseline --- random variable that is independent 
of action :math:`U_k` conditioned on observation :math:`Y_k`.


For Rewards
^^^^^^^^^^^
When optimizing for rewards, the general formula for the gradient update is:

.. math::
    \theta_{i+1} \leftarrow \theta_{i} + \alpha \mathbb{E}\left[ \sum_{k = 0}^{N-1} \left( \sum_{k'=k}^{N-1} \gamma^{k'}r(Y_{k'}, U_{k'}) - B_k \right)
    \nabla_{\theta} \ln\rho^{\theta}(U_k \mid Y_k)\big|_{\theta=\theta^i}\right].

For Costs
^^^^^^^^^
When optimizing for costs, the gradient update formula includes a negative sign:

.. math::
    \theta_{i+1} \leftarrow \theta_{i} - \alpha \mathbb{E}\left[ \sum_{k = 0}^{N-1} \left( \sum_{k'=k}^{N-1} \gamma^{k'}c(Y_{k'}, U_{k'}) - B_k \right)
    \nabla_{\theta} \ln\rho^{\theta}(U_k \mid Y_k)\big|_{\theta=\theta^i}\right].

Algorithm Steps
---------------

1. Initialize the baseline :math:`B^1 = 0`.
2. For each iteration :math:`i` in :math:`\{1, \ldots, N\_iterations\}`:

   a. For each episode :math:`j` in :math:`\{1, \ldots, M\}`:

      i. For each time step :math:`k` in :math:`\{0, \ldots, N-1\}`:

         - Obtain observation :math:`y_k^j` from the environment.
         - Sample action :math:`u_k^j` from a truncated normal distribution that is truncated to action bounds (provided as default parameters), using the policy :math:`\rho^{\theta}(u_k^j | y_k^j)`.

   b. Compute the baseline for the next iteration:

      .. math::
          b^{i + 1}_k \leftarrow \frac{1}{M} \sum_{j=1}^M \sum_{k'=k}^{N-1} \gamma^{k'} r(y_{k'}^j, u_{k'}^j)

   c. Perform a gradient step:

      .. math::
          \theta_{i+1} \leftarrow \theta_i - \alpha \frac{1}{M} \sum_{j=1}^{M} \sum_{k=0}^{N-1} \left(\sum_{k'=k}^{N-1} \gamma^{k'}
          c(y_{k'}^j, u_{k'}^j) - b^i_k\right)\nabla_\theta \ln\rho^\theta(u_k^j | y_k^j)\big|_{\theta=\theta^i}

Notation
--------

- :math:`N\_iterations`: Total number of iterations.
- :math:`M`: Number of episodes per iteration.
- :math:`N`: Number of steps per episode.
- :math:`\gamma`: Discount factor for future rewards.
- :math:`r(y_k^j, u_k^j)`: Reward received after taking action :math:`u_k^j` in state :math:`y_k^j`.
- :math:`c(Y_{k'}, U_{k'})`: Cost incurred after taking action :math:`U_{k'}` in state :math:`Y_{k'}`.
- :math:`\nabla_\theta`: Gradient with respect to policy parameters.
- :math:`\rho^\theta(u_k^j | y_k^j)`: Probability density function of the truncated normal distribution, representing the likelihood of taking action :math:`u_k^j` in state :math:`y_k^j` under the current policy parameterized by :math:`\theta`.



Usage Example (for costs)
-------------------------
.. code-block:: python

    import regelum as rg
    import numpy as np
    import torch
    from regelum.system import InvertedPendulumPD
    from regelum.objective import RunningObjective
    from regelum.simulator import CasADi
    from regelum.pipeline import ReinforcePipeline

    # Initialize the system and running cost
    system = InvertedPendulumPD()
    running_cost = RunningObjective(
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
            stds=[2.5],
            is_truncated_to_output_bounds=True,
        ),
        simulator=CasADi(
            system=system,
            state_init=np.array([[3.14, 0]]),
            time_final=10,
            max_step=0.001,
        ),
        policy_opt_method=torch.optim.Adam,
        # by default Adam minimizes objective, use {"lr": 0.1, "maximize": True} for rewards case
        policy_opt_method_kwargs={"lr": 0.1}, 
        running_objective=running_cost,
        sampling_time=0.01,
        N_episodes=4,
        N_iterations=100,
        is_with_baseline=True,
        is_do_not_let_the_past_distract_you=True,
    )

    # Run the training process
    pipeline.run()

Proximal Policy Optimization (PPO) Algorithm
============================================

Overview
--------
The PPOPipeline implements the Proximal Policy Optimization (PPO) algorithm, a policy gradient method for reinforcement learning that balances exploration and exploitation by clipping the policy update. PPO aims to take the biggest possible improvement step on a policy without causing performance collapse, thus ensuring a monotonic improvement.

Gradient Optimization Formulas
------------------------------

The optimization step in PPO is performed by solving a clipped surrogate objective function. The parameters for the policy are updated by minimizing the expected difference between the old and new policy while keeping the updates within a trust region.

For Policy Updates
^^^^^^^^^^^^^^^^^^
The policy parameters are updated using the following formula:

.. math::
    \theta_{i+1} \leftarrow \arg\min_{\theta}\mathbb{E}_{f, \rho^{\theta_i}}{\left[\sum_{k=0}^{N-1} \gamma^k \max\left(A^{\rho^{\theta_i}}(Y_k, U_k)\frac{\rho^{\theta}(U_k | Y_k)}{\rho^{\theta_i}(U_k | Y_k)}, A^{\rho^{\theta_i}}(Y_k, U_k) \operatorname{clip}_{1 - \varepsilon}^{1 + \varepsilon}\left(\frac{\rho^{\theta}(U_k | Y_k)}{\rho^{\theta_i}(U_k | Y_k)}\right) \right]\right)}.

where :math:`A^{\rho^{\theta_i}}(Y_k, U_k)` is the advantage function at timestep :math:`k`.

Algorithm Steps
---------------

1. Initialize policy parameters :math:`\theta^1` and baseline function :math:`J^w`.
2. For each iteration :math:`i \in \{1, \ldots, N\_iterations\}`:

   a. For each episode :math:`j \in \{1, \ldots, M\}`:

      i. For each time step :math:`k \in \{0, \ldots, N-1\}`:

         - Obtain observation :math:`y_k^j` from the environment.
         - Sample action :math:`u_k^j` from the policy :math:`\rho^{\theta}(u_k^j | y_k^j)`.
   
   b. Update the baseline function by fitting :math:`\hat{J}^{w}`.

   c. Update the policy by performing a policy gradient step using the clipped surrogate objective function.

Notation
--------

- :math:`N\_iterations`: Total number of training iterations.
- :math:`M`: Number of episodes per iteration.
- :math:`N`: Number of steps per episode.
- :math:`\gamma`: Discount factor for future rewards.
- :math:`A^{\rho^{\theta_i}}(Y_k, U_k)`: The advantage function, representing the relative value of action :math:`U_k` in state :math:`Y_k`.
- :math:`\rho^{\theta}(u_k^j | y_k^j)`: The policy distribution from which actions are sampled.
- :math:`\varepsilon`: The clipping hyperparameter that defines the trust region.


Proximal Policy Optimization Algorithm
======================================

General formula:

.. math::

  \boxed{
    \begin{array}{l}
    \hphantom{~}
    \\
    \theta_{i+1} \leftarrow \arg\min_{\theta}\mathbb{E}_{f, \rho^{\theta_i}}{\sum_{k=0}^{\infty} \gamma ^ k \max\left(A^{\rho^{\theta_i}}(Y_k, U_k)   \frac{\rho^{\theta}(U_k \mid Y_k)}{\rho^{\theta_i}(U_k \mid Y_k)}, A^{\rho^{\theta_i}}(Y_k, U_k) \operatorname{clip}_{1 - \varepsilon}^{1 + \varepsilon}\left(\frac{\rho^{\theta}(U_k \mid Y_k)}{\rho^{\theta_i}(U_k \mid Y_k)}\right) \right)}
    \\
    \hphantom{~}
    \end{array}
  }

where :math:`A^{\rho^{\theta_i}}(Y_k, U_k) = r(Y_k, U_k) + \gamma J^{\rho^{\theta_i}}(Y_{k+1}) - J^{\rho^{\theta_i}}(Y_{k})`.
On practice the number of steps :math:`k` is finite and we denote it by :math:`N`. The detailed description of algorithm is as follows 
(note that it works only for :math:`\gamma < 1`).

1. For each iteration :math:`i` in :math:`\{1, \ldots, N\_iterations\}`:

   a. For each episode :math:`j` in :math:`\{1, \ldots, M\}`:

      i. For each time step :math:`k` in :math:`\{0, \ldots, N-1\}`:

         - Obtain observation :math:`y_k^j` from the environment.
         - Sample action :math:`u_k^j` from a truncated normal distribution that is truncated to action bounds (provided as default parameters), using the policy :math:`\rho^{\theta}(u_k^j | y_k^j)`.

   b. Now we need to fit cost-to-go :math:`\hat{J}^{w}`. We can do it by minimizing temporal 
   difference loss with learning rate :math:`\eta` (note that :math:`N_{\text{TD}}`, :math:`N_{\text{epochs}}^{\text{Critic}}` 
   are also hyperparameters of algorithm). 
   The optimization procedure converges only for :math:`\gamma < 1`: 
   
      i. For each epoch in :math:`e` in :math:`\{1, \ldots,N_{\text{epochs}}^{\text{Critic}}`\}`
          
          - For each episode :math:`j` in :math:`\{1, \ldots, M\}`:       
            
              - :math:`w^{\text{new}} \leftarrow w^{\text{old}} - \eta \nabla_{w}\text{TDLoss}`

   c. Perform a policy gradient optimization procedure: For current policy weights :math:`\theta_{i}` 
   calculate :math:`\rho^{\theta_{i}}(u^j_k | y^j_k)` for all :math:`j`, :math:`k`
      
      i. For each epoch in :math:`e` in :math:`\{1, \ldots,N_{\text{epochs}}^{\text{Policy}}`\}`:
      
          -   :math:`\theta^{\text{new}}\leftarrow \theta^{\text{old}}-\alpha\nabla_{\theta}\text{PolicyObjective}`

In the listing above we denote :math:`\text{TDLoss}` as follows:

.. math::
    \frac{\sum\limits_{k = 0}^{N-1-N_{\text{TD}}} \left(\hat{J}^{w}\left(y^j_k\right) - r\left(y^j_k, u_k^j\right)  -... - \gamma^{N_{\text{TD}}-1} r\left(y^j_{k + N_{\text{TD}}-1}, u^j_{k + N_{\text{TD}}-1}\right) - \gamma^{N_{\text{TD}}} \hat{J}^{w}\left(y^j_{k + N_{\text{TD}}}\right)\right) ^ 2}{N-1-N_{\text{TD}}},

and :math:`\text{PolicyObjective}`:

.. math::
    \frac{1}{M}\sum\limits_{j=1}^{M}\sum\limits_{k=0}^{N-2}\gamma^k \max\left(\hat{A}^{w}(y^j_k, u^j_k)   \frac{\rho^{\theta}(u^j_k \mid y^j_k)}{\rho^{\theta_{i}}(u^j_k \mid y^j_k)}, \hat{A}^{w}(y^j_k, u^j_k) \operatorname{clip}_{1 - \varepsilon}^{1 + \varepsilon}\left(\frac{\rho^{\theta}(u^j_k \mid y^j_k)}{\rho^{\theta_{i}}(u^j_k \mid y^j_k)}\right) \right).

.. math::

    \newcommand{\clip}{clip_{1 - \varepsilon}^{1 + \varepsilon}}

    \clip + 1

.. math::

    \clip + 2

.. code-block:: python

    import regelum as rg
    import numpy as np
    import torch
    from regelum.system import InvertedPendulumPD
    from regelum.objective import RunningObjective
    from regelum.simulator import CasADi

    system = InvertedPendulumPD()
    pipeline = rg.pipeline.PPOPipeline(
        policy_model=rg.model.PerceptronWithTruncatedNormalNoise(
            dim_input=system.dim_observation,
            dim_hidden=4,
            n_hidden_layers=2,
            dim_output=system.dim_inputs,
            hidden_activation=torch.nn.LeakyReLU(0.2),
            output_activation=rg.model.MultiplyByConstant(1 / 100),
            output_bounds=system.action_bounds,
            stds=[[2.5]],
            is_truncated_to_output_bounds=True,
        ),
        critic_model=rg.model.ModelPerceptron(
            dim_input=system.dim_observation,
            dim_hidden=100,
            n_hidden_layers=4,
            dim_output=1,
        ),
        simulator=CasADi(
            system=system,
            state_init=np.array([[3.14, 0]]),
            time_final=10,
            max_step=0.001,
        ),
        critic_n_epochs=50,
        policy_n_epochs=50,
        critic_opt_method_kwargs={"lr": 0.0001},
        policy_opt_method_kwargs={"lr": 0.01},
        sampling_time=0.01,
        running_objective=RunningObjective(
            model=rg.model.ModelQuadLin("diagonal", weights=[10, 3.0, 0.0])
        ),
    )

    pipeline.run()

.. note::
    .. raw:: html
        :file: Osinenko2023habil.html

.. figure:: file.svg
   :width: 100%