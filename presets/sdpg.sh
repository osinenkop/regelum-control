python run_stable.py \
+seed=0 \
controller=sdpg \
scenario.N_episodes=2 \
scenario.N_iterations=200 \
system=3wrobot_ni,inv_pendulum,kin_point,lunar_lander,2tank \
--experiment=SDPG_GRID_SEARCH_2 \
scenario.discount_factor=0.7 \
--jobs=5
