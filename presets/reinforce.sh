python run_stable.py \
+seed=0 \
controller=reinforce \
scenario.N_episodes=3 \
scenario.N_iterations=500 \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=REINFORCE_GRID_SEARCH \
scenario.discount_factor=1.0 \
--jobs=-1