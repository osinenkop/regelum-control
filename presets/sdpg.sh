python run_stable.py \
+seed=0 \
controller=sdpg \
scenario.N_episodes=2 \
scenario.N_iterations=200 \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=SDPG_GRID_SEARCH_2 \
controller.discount_factor=0.7 \
--jobs=5
