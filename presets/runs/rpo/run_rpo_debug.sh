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
    initial_conditions=ic_${system}_stochastic \
    controller.actor.discount_factor=.99 \
    controller.critic.model.force_positive_def=false \
    controller.critic.data_buffer_size=500 \
    controller.critic.batch_size=30 \
    controller.critic.td_n=30 \
    simulator.time_final=3 \
    scenario.N_episodes=100
