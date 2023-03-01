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
    --cooldown-factor=8.0 \
    scenario.N_episodes=10 \
    +seed=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 \
    --single-thread

echo "Controller value: $controller"
