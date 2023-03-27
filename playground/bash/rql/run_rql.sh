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
    controller=rql \
    initial_conditions=ic_${system}_stochastic \
    controller.actor.predictor.prediction_horizon=6 \
    controller/critic=dqn_behaviour \
    controller.critic.model.force_positive_def=false \
    controller.critic.data_buffer_size=500 \
    controller.critic.batch_size=30 \
    controller.critic.td_n=30 \
    scenario.N_episodes=10
    +seed=1,2
