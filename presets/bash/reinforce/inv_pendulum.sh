python run_stable.py \
+seed=1,2,3,4,5,6,7,8,9,10 \
scenario=reinforce \
scenario.N_episodes=4 \
scenario.N_iterations=500 \
system=inv_pendulum \
--experiment=reinforce_inv_pendulum \
scenario/policy_model=perceptron_simple \
scenario.policy_n_epochs=1 \
scenario.policy_opt_method_kwargs.lr=0.1 \
scenario.policy_model.n_hidden_layers=1 \
scenario.policy_model.dim_hidden=4 \
scenario.policy_model.leaky_relu_slope=0.2 \
scenario.policy_model.normalize_output_coef=0.005 \
--jobs=-1
