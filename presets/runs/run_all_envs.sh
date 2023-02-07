#!/bin/bash

controller=$1

for system in 2tank 3wrobot_ni 3wrobot cartpole inv_pendulum kin_point lunar_lander
do
    tmux new-session -d -s "${controller}-${system}"
    tmux send-keys -t "${controller}-${system}" "source ../env/bin/activate" ENTER
    tmux send-keys -t "${controller}-${system}" "bash runs/${controller}/run_${controller}.sh true ${system}" ENTER
    echo "Created tmux session ${controller}-${system}"
    sleep 2
done
