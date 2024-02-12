python run_stable.py \
+seed=0,1,2,3,4,5,6,7,8,9 \
--jobs=-1 \
scenario=sdpg \
system=2tank \
--experiment=sdpg_2tank \
scenario.N_episodes=4 \
scenario.N_iterations=500 \
scenario/policy_model=perceptron_simple \
scenario.critic_n_epochs=50 \
scenario.critic_model.n_hidden_layers=2 \
scenario.critic_model.dim_hidden=15 \
scenario.critic_opt_method_kwargs.lr=0.1 \
scenario.discount_factor=0.99 \
--jobs=-1