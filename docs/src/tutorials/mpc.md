

Common AIDA notation
====================

  -------------------------- -------------------------------------------
  $\R_{\ge 0}, \Z_{\ge 0}$   Set of nonnegative reals, resp., integers
  $\PP{\bullet}$             Probability measure
  $\E[f]{\bullet}$           Expectation under distribution $f$
  $\nrm{\bullet}$            Norm (context-dependent)
  -------------------------- -------------------------------------------

Reinforcement learning and control notation.
--------------------------------------------

  -------- ---------------------------------------------------------------------
  $Q$      Quality function
  $r, R$   Running objective (reward or cost) as definite, resp., random value
  -------- ---------------------------------------------------------------------

#### Conditional notation.

When `mlnotation` toggle is set `true`. You should set the toggle in
`PREAMBLE` section right after loading the preamble.

  ----------------------------- ------------------------------------------------------------------------------
  $\policy$                     Policy, a law that generates actions from observations
  $\state, \State$              State, as definite, resp., random value
  $\action, \Action$            Action, as definite, resp., random value
  $\obs, \Obs$                  Observation, as definite, resp., random value
  $\states$                     State space
  $\actions$                    Action space
  $\obses$                      Observation space
  $\policies$                   Policy space
  $\transit$                    State dynamics law (function or probability distribution)
  $\Value^\policy$              Total objective (value or cost) of policy $\policy$
  $\Value^*$                    Optimum total objective (value or cost) under the optimal policy $\policy^*$
  $\Advan^{\policy,\policy'}$   Advantage of policy $\policy$ relative to policy $\policy'$
  $\policy^\theta$              Actor network with weights $\theta$
  $\hat \Value^w$               Critic network (state-valued) with weights $w$
  ----------------------------- ------------------------------------------------------------------------------

Introduction
============

Reinforcement learning commonly addresses the following infinite-horizon
optimal control and/or decision problem: $$\label{eqn_optctrl_problem}
    \begin{aligned}
    & \, \max_{\policy \in \policies} \; \Value^\policy (\state) = \\
    & \, \max_{\policy \in \policies} \; \E[\Action_t \sim \policy]{\sum\limits_{t=0}^{\infty} \gamma^t r(\State_t, \Action_t) \vert \State_0=\state},
    \end{aligned}$$ where $\State_t \in \states$ at a time
$t \in \T := \Z_{\ge0}$ is the environment's state with values in the
state-space $\states$, $r$ is the reward (in general, running objective)
rate, $\gamma$ is the discount factor, $\policy$ is the agent's policy
of some function class $\policies$. The running objective may be taken
as a random variable $R_t$ whose probability distribution depends on the
state and action. The agent-environment[^1] loop dynamics are commonly
modeled via the following Markov chain: $$\label{eqn_sysmarkov}
    \begin{aligned}
        & \State_{t+1} \sim \transit(\bullet \vert \state_t, \action_t), \spc t \in \T.
    \end{aligned}$$

For the problem
[\[eqn\_optctrl\_problem\]](#eqn_optctrl_problem){reference-type="eqref"
reference="eqn_optctrl_problem"}, one can state an important recursive
property of the objective optimum $\Value^*(\state)$ in the form of the
Hamilton-Jacobi-Bellman (HJB) equation as follows: $$\label{eqn_hjb}
    \max_{\action \in \actions}{\{ \mathcal D^\action \Value^*(\state) + r(\state, \action) - \gamma \Value^*(\state)\}} = 0, \forall \state \in \states,$$
where
$\mathcal D^\action \Value^*(\state) := \E[S_+ \sim \transit(\bullet \vert \state, \action)]{\Value^*((\State_{+}))} - \Value^*(\state)$.

The common approaches to
[\[eqn\_optctrl\_problem\]](#eqn_optctrl_problem){reference-type="eqref"
reference="eqn_optctrl_problem"} are dynamic programming
[@Bertsekas2019Reinforcementl; @Lewis2009Reinforcementl] and
model-predictive control
[@Garcia1989Modelpredictiv; @Borrelli2011PredictiveCont; @Darby2012MPCCurrentpra; @Mayne2014Modelpredictiv].
The latter cuts the infinite horizon to some finite value $T>0$ thus
considering effectively a finite-time optimal control problem. Dynamic
programming aims directly at the HJB
[\[eqn\_hjb\]](#eqn_hjb){reference-type="eqref" reference="eqn_hjb"} and
solves it iteratively over a mesh in the state space $\states$ and thus
belongs to the category of tabular methods. The most significant problem
with such a discretization is the curse of dimensionality, since the
number of nodes in the said mesh grows exponentially with the dimension
of the state space. Evidently, dynamic programming is in general only
applicable when the state-space is compact. Furthermore, state-space
discretization should be fine enough to avoid undesirable effects that
may lead to a loss of stability of the agent-environment closed loop.
Reinforcement learning essentially approximates the optimum objective
$\Value^*$ via a (deep) neural network.

$\theta_0$ Policy weight update
$\theta_{i+1} \la \theta_i - \alpha_i \Es[\trajpdf^{\policy^{\theta_i}}]{ \Cost^\gamma_{0:T} \sum \limits_{t=0}^{T-1} \nabla_\theta \log \policy^{\theta_i} ( \Traj_t ) }$
$\alpha_i > 0$, learning rate Near-optimal policy
$\policy^{\theta_{\mathcal I}}$

[^1]: In classical terminology, "controller-system" loop