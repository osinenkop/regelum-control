#!/bin/bash

bash runs/mpc/run_mpc.sh true 2tank
bash runs/mpc/run_mpc.sh true 3wrobot_ni
bash runs/mpc/run_mpc.sh true 3wrobot
bash runs/mpc/run_mpc.sh true cartpole
bash runs/mpc/run_mpc.sh true inv_pendulum
bash runs/mpc/run_mpc.sh true kin_point
bash runs/mpc/run_mpc.sh true lunar_lander