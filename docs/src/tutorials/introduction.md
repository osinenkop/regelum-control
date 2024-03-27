# Introduction to Regelum: Python Framework for Reinforcement Learning and Optimal Control

Welcome to the tutorial on Regelum, a Python framework for reinforcement learning and optimal control. In this tutorial, we will cover all the key concepts and functionality of the framework, providing you with a step-by-step guide to understanding and utilizing Regelum effectively.

## Who is this tutorial for?

This tutorial is designed for engineers, researchers, and enthusiasts who are interested in reinforcement learning and optimal control techniques. Whether you are a seasoned professional or just starting your journey in this field, this tutorial will provide you with the knowledge and practical skills needed to leverage Regelum for solving complex control problems.

## What is Regelum?

Regelum is a versatile framework that combines the principles of reinforcement learning and optimal control to enable you to design and implement sophisticated control strategies. It provides a flexible interface for developing and deploying control algorithms, making it easier for you to experiment, iterate, and optimize your control systems.

## Tutorial Structure

To ensure a smooth learning experience, this tutorial follows a linear structure, gradually introducing you to the key ideas and concepts of the Regelum framework. Here is an overview of the tutorial sections:

1. [Introduction](./introduction.md): This section (you're currently reading) provides an overview of Regelum and sets the stage for the rest of the tutorial.

2. [Basic Concepts](./basic_concepts.md): In this section, we will explore the fundamental concepts underlying reinforcement learning and optimal control. This includes an introduction to state, action, reward, and the dynamics of a system.

3. [Implementing your own dynamical system](system.md): Here, we will dive into the practical aspects of implementing a dynamical system within the Regelum framework. You will learn how to define the state and action spaces, as well as the dynamics of your system.

4. [Run your system in simulator](../notebooks/simulator.md): This section focuses on using simulators to evaluate the performance of your control system. We will guide you through the process of integrating your system with a simulator and running simulations.

5. [Run the scenario with a custom policy](../notebooks/policy.md): The policy is a crucial component in reinforcement learning. In this section, you will learn how to define and train policies using Regelum, enabling your control system to make intelligent decisions.

6. [Optimizable and models](../notebooks/optimizable.md)

7. [Model Predictive Control](../notebooks/mpc.md): MPC is a powerful technique for optimizing control actions in real-time. Here, we will explore how to implement MPC using Regelum, enabling your system to adapt and respond to changing environments.

8. [Policy gradient](../notebooks/policy_gradient.md): In this tutorial we will implement a well-known policy gradient algorithm Vanilla Policy Gradient with General Advantage Estimation. Moreover, we will create a complex system composed of two different kinematic systems, namely, three wheeled robot and a kinematic point.
Appears that this perfectly represents an environment from a well-known problem, namely, homicidal chauffeur problem.

9. [Callbacks](../notebooks/callbacks.md): Advantage-Actor Critic with Generalized Policy Gradient (ACPG) combines value estimation with policy optimization. In this section, we will explore how to incorporate a critic into your control system using Regelum, enhancing its decision-making capabilities.

10. [Configuring your experiment](configs.md): In this section, we will guide you on how to configure and customize your control experiments in Regelum. 

11. [Stable presets](stable-presets.md): Access and experiment with our pre-configurations of RL and Optimal Control alorithms.

12. [Animations](animations.md): A comprehensive guide about creating animations for your own dynamical systems. 

By the end of this tutorial, you will have a solid understanding of the Regelum framework and be equipped with the skills to apply reinforcement learning and optimal control techniques to solve real-world control problems.

Happy learning, and may your code run error-free!