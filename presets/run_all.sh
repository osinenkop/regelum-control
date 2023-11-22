python run_stable.py \
+seed=5,6 \
scenario=ppo \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=SCENARIO_PPO \
--jobs=-1

python run_stable.py \
+seed=5,6 \
scenario=sdpg \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=SCENARIO_SDPG \
--jobs=-1 


python run_stable.py \
+seed=5,6 \
scenario=ddpg \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=SCENARIO_DDPG \
--jobs=-1

python run_stable.py \
+seed=5,6 \
scenario=reinforce \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=SCENARIO_REINFORCE \
--jobs=-1
