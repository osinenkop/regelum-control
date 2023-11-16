python run_stable.py \
+seed=0 \
pipeline=ppo \
scenario.N_episodes=2 \
scenario.N_iterations=200 \
system=inv_pendulum \
--experiment=PPO_GRID_SEARCH \
pipeline.discount_factor=0.7 \
pipeline/policy/model=perceptron_with_truncated_normal_noise \
--jobs=-1 
