python run_stable.py \
+seed=0 \
controller=ppo \
scenario.N_episodes=2 \
scenario.N_iterations=200 \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=PPO_GRID_SEARCH \
scenario.discount_factor=0.7 \
--jobs=-1 
