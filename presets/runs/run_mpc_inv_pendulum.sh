#!/bin/bash

disallow_uncommitted=$1

if [[ $disallow_uncommitted = false ]]
then
    disallow_uncommitted="false"
else
    disallow_uncommitted="true"
fi

parentdir=$(dirname $PWD)

PYTHONPATH=$parentdir python preset_endpoint.py disallow_uncommitted=$disallow_uncommitted scenario.is_playback=false \
system=inv_pendulum \
controller=mpc \
initial_conditions=ic_inv_pendulum_stochastic \
+seed=1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
controller.actor.predictor.prediction_horizon=2

PYTHONPATH=$parentdir python preset_endpoint.py disallow_uncommitted=$disallow_uncommitted scenario.is_playback=false \
system=inv_pendulum \
controller=mpc \
initial_conditions=ic_inv_pendulum_stochastic \
+seed=1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
controller.actor.predictor.prediction_horizon=5

PYTHONPATH=$parentdir python preset_endpoint.py disallow_uncommitted=$disallow_uncommitted scenario.is_playback=false \
system=inv_pendulum \
controller=mpc \
initial_conditions=ic_inv_pendulum_stochastic \
+seed=1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
controller.actor.predictor.prediction_horizon=8

