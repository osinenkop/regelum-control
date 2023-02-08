#!/bin/bash

disallow_uncommitted=$1
system=$2

if [ "$disallow_uncommitted" == "false" ];
then
    disallow_uncommitted="false"
else
    disallow_uncommitted="true"
fi

parentdir=$(dirname $PWD)

PYTHONPATH=$parentdir python preset_endpoint.py disallow_uncommitted=$disallow_uncommitted scenario.is_playback=false --cooldown-factor=8.0 \
    system=$system \
    controller=dqn \
    initial_conditions=ic_${system}_stochastic \
    scenario.N_episodes=100 \
    controller.actor.discount_factor=.99 \
    controller.critic.model.dim_hidden=40 \
    controller.critic.optimizer.opt_options.lr=0.001 \
    controller.critic.optimizer.iterations=1 \
    controller.critic.model.force_positive_def=false \
    controller.critic.data_buffer_size=500 \
    controller.critic.batch_size=30 \
    controller.critic.td_n=30 \
    controller.critic.model.bias=true \
    controller/critic=dqn_greedy \
    controller.actor.epsilon_greedy=true \
    +seed=1
    cat
