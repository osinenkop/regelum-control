#!/bin/bash

controller=$1
seed_from=$2
seed_to=$3
override=$4
override2=$5

for system in 2tank 3wrobot_ni 3wrobot cartpole inv_pendulum kin_point lunar_lander
do
    tmux new-session -d -s "${controller}-${system}"
    tmux send-keys -t "${controller}-${system}" "source ../env/bin/activate" ENTER
    tmux send-keys -t "${controller}-${system}" "bash runs/seeds.sh ${controller} ${system} ${seed_from} ${seed_to} ${override} ${override2}" ENTER
    echo "Created tmux session ${controller}-${system}"
done
