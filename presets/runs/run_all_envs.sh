#!/bin/bash

controller=$1
override=$2

for system in 2tank 3wrobot_ni 3wrobot cartpole inv_pendulum kin_point lunar_lander
do
    tmux new-session -d -s "${controller}-${system}"
    tmux send-keys -t "${controller}-${system}" "source ../env/bin/activate" ENTER
    tmux send-keys -t "${controller}-${system}" "bash runs/seeds.sh false ${controller} ${system} ${override}" ENTER
    echo "Created tmux session ${controller}-${system}"
done
