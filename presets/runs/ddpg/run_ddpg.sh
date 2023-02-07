#!/bin/bash

disallow_uncommitted="false"
system=$1

if [ "$disallow_uncommitted" == "false" ];
then
    disallow_uncommitted="false"
else
    disallow_uncommitted="true"
fi

parentdir=$(dirname $PWD)

PYTHONPATH=$parentdir python preset_endpoint.py disallow_uncommitted=$disallow_uncommitted scenario.is_playback=false --cooldown-factor=8.0 \
    system=$system \
    controller=ddpg \
    scenario=episodic_reinforce \
    scenario.N_episodes=20 \
    initial_conditions=ic_${system}_stochastic \
    controller.actor.discount_factor=.99 \
    controller.critic.model.force_positive_def=false \
    controller.critic.data_buffer_size=100 \
    controller/critic=action_observation_on_policy
   # +seed=1,2
