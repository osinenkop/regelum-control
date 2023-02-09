#!/bin/bash

controller=$1
system=$2
seed=$3
override=$4

parentdir=$(dirname $PWD)

PYTHONPATH=$parentdir python preset_endpoint.py $override \
    controller=$controller \
    system=$system \
    scenario.is_playback=false \
    --cooldown-factor=8.0 \
    +seed=$seed 