python run_stable.py \
+seed=1,2,3,4,5,6,7,8,9,10 \
--jobs=-1 \
scenario=reinforce \
system=2tank \
--experiment=reinforce_2tank \
scenario.N_episodes=4 \
scenario.N_iterations=300 \
scenario/policy_model=perceptron_simple