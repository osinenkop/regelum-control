#!/bin/bash

system=$2


if [ "$1" == "calf" ] && [ "$2" == "cartpole" ]; then
  controller="calf_predictive"
else
  controller="calf_ex_post"
fi


python preset_endpoint.py \
    controller=$controller \
    system=$system \
    scenario.is_playback=false \
    --cooldown-factor=2.0 \
    scenario.N_episodes=6 \
    +seed=1 \
    --single-thread

echo "Controller value: $controller"