#!/bin/bash

disallow_uncommitted=$1
controller=$2
system=$3
seed=$4
override=$5

if [ "$disallow_uncommitted" == "false" ];
then
    disallow_uncommitted="false"
else
    disallow_uncommitted="true"
fi

parentdir=$(dirname $PWD)

PYTHONPATH=$parentdir python preset_endpoint.py $override \
    controller=$controller \
    system=$system \
    disallow_uncommitted=true \
    scenario.is_playback=false \
    --cooldown-factor=8.0 \
    +seed=$seed 