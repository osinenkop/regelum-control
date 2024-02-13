## What is a Scenario?

A `Scenario` in the Regelum framework serves as an orchestration layer for reinforcement learning and optimal control problems. It manages the simulation of the agent-environment interactions by driving the agent's decision-making process based on the defined `Policy`. A scenario runs a sequence of actions and observations over time, handles the agent's performance assessment, and optionally incorporates optimization techniques for policy improvement.

Scenarios are fundamental in setting up and running experiments which are crucial for training and evaluating agents under different conditions. They ensure consistency in how agents are tested and benchmarked.

## Types of Scenarios in Regelum

There 2 main types of scenarios in regelum: 

1. [`Scenario`][regelum.scenario.Scenario]:
    - It usually involves a simple policy that maps observations or states directly to actions.
    - The agent follows the policy to perform actions in the environment and receives immediate feedback through the simulator.
    - There is no explicit learning or optimization process involved.

2. [`RLScenario`][regelum.scenario.RLScenario]:
    - This more advanced type allows for policy optimization based on the agent's experience.
    - This type typically involves a [`Critic`][regelum.critic.Critic] that evaluates the quality or value of the chosen actions or the states visited by the agent.
    - The critic provides additional feedback to the agent, which is typically used to optimize the policy.
    - The critic can be a simple value function or a complex neural network.
    - May implement algorithms such as REINFORCE, Proximal Policy Optimization (PPO), DQN, etc.

    !!! Note
        [**RLScenario**][regelum.scenario.RLScenario] supports integration with a [simple mock critic][regelum.critic.CriticTrivial], which is applicable for such scenarios as Model Predictive Control. 

TODO: Add link to MPC

## Episode vs. Iteration

The classes [Scenario][regelum.scenario.Scenario] and [RLScenario][regelum.scenario.RLScenario] accept two key parameters upon instantiation:

- Number of **episodes**. Below the provide the detailed explanation of what is the **episode**:
    
    - An **episode** represents one complete sequence of the agent's interaction with the environment, from an initial state to a terminal state or until a certain stopping condition is met (for example, a time limit).
    
    - **Episodes** are useful to simulate and learn from diverse situations by resetting the environment to various initial states.
    
    - **Episodes** allow the agent to experience the consequences of its actions in various environmental conditions, contributing to its overall learning process.

- Number of **iterations**. Below the provide the detailed explanation of what is the **iteration**:
    
    - An **iteration** encompasses one or more episodes.

    - During an **iteration**, the agent may update its policy through learning, with the goal of improving its performance over subsequent episodes.

    - **Iterations** enable the agent to learn incrementally by processing the experiences acquired across multiple episodes within the same or separate iterations, refining the policy after each iteration.

    - In some contexts, an **iteration** might correspond to a training epoch in machine learning, where the policy is optimized using data aggregated over several episodes.

    - Iterations provide a framework for the agent's development over time, allowing for sustained learning and policy improvement.

## What is a Policy?

In reinforcement learning and optimal control within the Regelum framework, a `Policy` defines the strategy or rules by which an agent decides to take actions in an environment based on its observations. 
It encapsulates the decision-making process that maps observations or states of the environment to specific actions with the goal of achieving a certain objective, such as 
- maximizing cumulative rewards
- minimizing cumulative costs

Policies play a crucial role in the learning process as they directly influence the agent's behavior and its ability to solve the given task. 
They can range from simple rule-based strategies to complex neural network models that require training through interaction with the environment.

## Why Use a Policy?

- **Action Selection**: The policy provides a systematic way for the agent to select actions based on the current state or observation from the environment.
- **Learning and Adaptation**: Through interaction with the environment, the policy can be optimized to improve its performance over time, allowing the agent to learn and adapt.
- **Goal Achievement**: A well-defined policy enables the agent to perform tasks that achieve desired goals, such as reaching a target location, maintaining stability, or maximizing rewards.
