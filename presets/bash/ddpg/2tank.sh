python run_stable.py \
+seed=1,2,3,4,5,6,7,8,9,10 \
scenario=ddpg \
system=2tank \
--experiment=ddpg_2tank \
scenario.N_episodes=6 \
scenario.is_parallel=False \
scenario.N_iterations=50 \
scenario.critic_opt_method_kwargs.lr=0.01 \
scenario.critic_n_epochs=1 \
scenario.policy_opt_method_kwargs.lr=0.01 \
scenario.policy_n_epochs=1 \
scenario.policy_model.n_hidden_layers=1 \
scenario.policy_model.dim_hidden=20 \
scenario.critic_model.n_hidden_layers=2 \
scenario.critic_model.dim_hidden=40 \
+scenario.critic_model.is_force_infinitesimal=True \
scenario.discount_factor=0.75
