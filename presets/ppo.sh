python run_stable.py \
+seed=0 \
controller=ppo \
scenario.N_episodes=2 \
scenario.N_iterations=200 \
system=inv_pendulum \
--experiment=PPO_GRID_SEARCH \
scenario.discount_factor=0.7 \
controller/policy/model=perceptron_with_truncated_normal_noise \
--jobs=-1 
