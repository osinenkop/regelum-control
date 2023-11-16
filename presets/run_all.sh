python run_stable.py \
+seed=5,6 \
pipeline=ppo \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=PPO_TRUNCATED_vs_NORMAL \
--jobs=-1

python run_stable.py \
+seed=5,6 \
pipeline=sdpg \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=SDPG_TRUNCATED_vs_NORMAL \
--jobs=-1 


python run_stable.py \
+seed=5,6 \
pipeline=ddpg \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=DDPG_TRUNCATED_vs_NORMAL \
--jobs=-1

python run_stable.py \
+seed=5,6 \
pipeline=reinforce \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=REINFORCE_TRUNCATED_vs_NORMAL \
pipeline.discount_factor=1.0 \
--jobs=-1
