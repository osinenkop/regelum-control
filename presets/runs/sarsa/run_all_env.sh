#!/bin/bash

for system in 2tank 3wrobot_ni 3wrobot cartpole inv_pendulum kin_point lunar_lander
do
    echo tmux new-session -s "sarsa-${system}" -d "bash runs/sarsa/run_sarsa.sh true ${system}"
done
#tmux new-session -d -s "" /opt/my_script.sh
