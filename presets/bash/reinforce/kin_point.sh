python run_stable.py \
+seed=1,2,3,4,5,6,7,8,9,10 \
--jobs=-1 \
scenario=reinforce \
system=kin_point \
--experiment=reinforce_kin_point \
scenario.N_episodes=4 \
scenario.N_iterations=300 \
scenario/policy_model=perceptron_simple \
scenario.policy_model.leaky_relu_slope=0.15