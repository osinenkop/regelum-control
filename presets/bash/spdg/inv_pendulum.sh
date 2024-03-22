python run_stable.py \
    +seed=range\(10\) \
    +n_jobs=-1 \
    scenario=sdpg \
    system=inv_pendulum \
    --experiment=sdpg_inv_pendulum \
    scenario.N_episodes=3 \
    scenario.is_parallel=False \
    scenario.N_iterations=100 \
    scenario.policy_opt_method_kwargs.lr=0.01 \
    scenario.policy_model.n_hidden_layers=2 \
    scenario.policy_model.dim_hidden=15 \
    scenario.policy_model.std=0.1 \
    scenario.critic_model.n_hidden_layers=2 \
    scenario.critic_model.dim_hidden=15 \
    scenario.critic_opt_method_kwargs.lr=0.1 \
    scenario.gae_lambda=0.96
