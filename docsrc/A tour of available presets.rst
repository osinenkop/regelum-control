A tour of available presets
===========================

This document provides run commands for several algorithms
 - Model Predictive Control
 - CALF (with safe stabilizing policies)
 - RL algorithms

on different environments also known as systems. It can be done via running ``preset_endpoint.py`` with different flags. 
``preset_endpoint.py`` works as follows. It parses provided flags and configs that are located in general folder with further instantiation
the scenario class (and corresponding system animator if necessary) and runs the learning procedure. For detailed explanation of how to work
with configuration files in rcognita go to this page. The list of available systems can be found here. And their corresponding instantiation 
configurations are located in this folder. Note that we used simplified notation in the provided configs (link)

So, the list of available systems that are provided in configuration files is the following:
 - ``2tank`` that instatiates rcognita.systems.System2Tank
 - ``3wrobot_ni`` that instatiates rcognita.systems.Sys3WRobotNI
 - ``3wrobot`` that instatiates rcognita.systems.Sys3WRobot
 - ``cartpole`` that instatiates rcognita.systems.SysCartpole
 - ``inv_pendulum`` that instatiates rcognita.systems.SysInvertedPedulum
 - ``kin_point`` that instatiates rcognita.systems.SysKinematicPoint
 - ``lunar_lander`` that instatiates rcognita.systems.SysLunarLande

Environment variables setup
---------------------------   
In the root of the repo run the following commands
::

    export HYDRA_FULL_ERROR=1
    export PYTHONPATH=$(pwd)

Playground
-----------

For quick start go to playground via ``cd playground`` and run
::

    python preset_endpoint.py system=3wrobot_ni controller=mpc cenario.howanim=live animator.fps=100 scenario.N_episodes=1

and see how the Model Predictive Control is stabilizing 3wrobot_mi system in live animation. If you want to turn off the animation just delete the flags 
``scenario.howanim=live animator.fps=100``: 
:: 

    python preset_endpoint.py controller=mpc system=3wrobot scenario.N_episodes=1

This command run MPC algorithm for stabilizing of 3wrobot_ni system. If you want to stabilize another system, for instance, inverted pendulum, you 
should override ``system=`` flag like this:
::

    python preset_endpoint.py controller=mpc system=inv_pendulum scenario.N_episodes=1

The list of available systems is the following: ``2tank``, ``3wrobot_ni``, ``3wrobot``, ``cartpole``. ``inv_pendulum``, ``kin_point``, ``lunar_lander``.

To run CALF alogorithm just write
:: 

    python preset_endpoint.py controller=calf_ex_post system=3wrobot scenario.N_episodes=10

This command runs CALF algorithm for stabilizing of 3wrobot system. Again, if you want to stabilize another system just override ``system=`` flag. To run 
the CALF stabilizing policy one should use flag ``+controller.safe_only=True`` with ``scenario.N_episodes=1``

:: 

    python preset_endpoint.py controller=calf_ex_post system=3wrobot scenario.N_episodes=1 +controller.safe_only=True



Useful Flags
------------

 - ``scenario.howanim=live`` - for live animation
 - ``scenario.howanim=playback`` - for animation playback without saving
 - ``scenario.howanim=html`` - for saving animation in html
 - ``scenario.howanim=mp4`` - for saving animation in mp4
 - ``scenario.howanim=None`` - do not activate animation at all
 - ``scenario.N_episodes=2`` - run scenario with 2 episodes
 - ``simulator.time_final=10`` - set time_final 
 - ``animator.fps=20`` - use this parameter for changing fps for animation
 - ``system_specific.sampling_time=0.03`` - set the sampling time
 - ``controller=dqn`` - possible variants are ``[dqn,mpc,calf_ex_post,calf_predictive,rql,sarsa,ddqn,sql,ddqn,acpg,ddpg]``
 - ``--cooldown-factor=0.1`` - decrease all cooldowns 10-fold
 - ``system=2tank`` - possible variants are ``[2tank,3wrobot,3wrobot_ni,cartpole,inv_pendulum,kin_point,lunar_lander]``
 - ``system_specific.calf_data_buffer_size=200``--- change calf buffer to 200
 - ``system=inv_pendulum controller=mpc controller.actor.predictor.prediction_horizon=3`` --- run mpc with prediction horizon 3