## Problem statement version 1


Reinforcement learning deals with a special kind of Markov Decision Problems (MDPs) which have a normed vector space as a state-space:

$$
\left(\states, \actions, \transit, \Cost \right),
$$

where:

- $\states$ is the *state space*, that is a normed vector space of all states of the given environment;

- $\actions$ is the *action space*, that is a set of all actions available to the agent;

- $\transit : \states \times \actions \times \states \ \rightarrow \ \R$ is the *transition probability density function* of the environment, that is such function that $\transit(\cdot \mid \state_{t}, \action_{t})$ is the probability density of the next state $s_{t + 1}$ conditioned on the current state $\state_{t}$ and current action $\action_{t}$;

- $\Cost : \states \times \actions \rightarrow \R$ is the *cost function* of the environment, that is a function that takes a state $\state_{t}$ and an action $\action_{t}$ and returns the immediate cost $\cost_{t}$ incurred upon the agent if it were to perform action $\action_{t}$ while in state $\state_{t}$;

**The goal of reinforcement learning** is to find a policy $\policy$ that minimizes 

$$
    V^{\policy}(\state_0) = \E{\sum_{t = 0}^{\infty}\gamma^{t}\Cost(\state_{t}, \action_{t})} \text{ for some } \gamma \in (0, 1].
$$ 

The policy $\policy^{\ast}$ that solves this problem is commonly referred to as *the optimal policy*.

## Problem statement version 2

Reinforcement learning commonly addresses the following infinite-horizon optimal control and/or decision problem:

\begin{equation}
	\label{eqn_optctrl_problem}
	\begin{aligned}
	& \, \max_{\policy \in \policies} \; \Value^\policy (\state) = \\
	& \, \max_{\policy \in \policies} \; \E[\Action_t \sim \policy]{\sum\limits_{t=0}^{\infty} \gamma^t r(\State_t, \Action_t) \vert \State_0=\state},
	\end{aligned}
\end{equation}

where $\State_t \in \states$ at a time $t \in \T := \Z_{\ge0}$ is the environment's state with values in the state-space $\states$, $r$ is the reward (in general, running objective) rate, $\gamma$ is the discount factor, $\policy$ is the agent's policy of some function class $\policies$.
The running objective may be taken as a random variable $R_t$ whose probability distribution depends on the state and action.
The agent-environment loop (or in classical terminology, "controller-system" loop) dynamics are commonly modeled via the following Markov chain:

\begin{equation}
	\label{eqn_sysmarkov}
	\begin{aligned}
		& \State_{t+1} \sim \transit(\bullet \vert \state_t, \action_t), \spc t \in \T.
	\end{aligned}
\end{equation}

For the problem $\eqref{eqn_optctrl_problem}$, one can state an important recursive property of the objective optimum $\Value^*(\state)$ in the form of the Hamilton-Jacobi-Bellman (HJB) equation as follows:

\begin{equation}
    \label{eqn_hjb}
    \max_{\action \in \actions}{\{ \mathcal D^\action \Value^*(\state) + r(\state, \action) - \gamma \Value^*(\state)\}} = 0, \forall \state \in \states,
\end{equation}

where $\mathcal D^\action \Value^*(\state) := \E[S_+ \sim \transit(\bullet \vert \state, \action)]{\Value^*((\State_{+}))} - \Value^*(\state)$.

The common approaches to $\eqref{eqn_optctrl_problem}$ are dynamic programming [@Bertsekas2019Reinforcementl;@Lewis2009Reinforcementl] and model-predictive control [@Garcia1989Modelpredictiv;@Borrelli2011PredictiveCont;@Darby2012MPCCurrentpra;@Mayne2014Modelpredictiv].
The latter cuts the infinite horizon to some finite value $T>0$ thus considering effectively a finite-time optimal control problem.
Dynamic programming aims directly at the HJB $\eqref{eqn_hjb}$ and solves it iteratively over a mesh in the state space $\states$ and thus belongs to the category of tabular methods.
The most significant problem with such a discretization is the curse of dimensionality, since the number of nodes in the said mesh grows exponentially with the dimension of the state space.
Evidently, dynamic programming is in general only applicable when the state-space is compact.
Furthermore, state-space discretization should be fine enough to avoid undesirable effects that may lead to a loss of stability of the agent-environment closed loop.
Reinforcement learning essentially approximates the optimum objective $\Value^*$ via a (deep) neural network.

## Notation

| Symbol                      | Description                                                                  |
|-----------------------------|------------------------------------------------------------------------------|
| $\policy$                   | Policy, a law that generates actions from observations                       |
| $\state, \State$            | State, as definite, resp., random value                                      |
| $\action, \Action$          | Action, as definite, resp., random value                                     |
| $\obs, \Obs$                | Observation, as definite, resp., random value                                |
| $\states$                   | State space                                                                  |
| $\actions$                  | Action space                                                                 |
| $\obses$                    | Observation space                                                            |
| $\policies$                 | Policy space                                                                 |
| $\transit$                  | State dynamics law (function or probability distribution)                    |
| $\Value^\policy$            | Total objective (value or cost) of policy $\policy$                          |
| $\Value^*$                  | Optimum total objective (value or cost) under the optimal policy $\policy^*$ |
| $\Advan^{\policy,\policy'}$ | Advantage of policy $\policy$ relative to policy $\policy'$                  |
| $\policy^\theta$            | Actor network with weights $\theta$                                          |
| $\hat \Value^w$             | Critic network (state-valued) with weights $w$                               |

## Bibliography
<!-- rendered automatically -->