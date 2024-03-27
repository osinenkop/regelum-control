python run_stable.py \
    --jobs=-1 \
    +seed=0,1,2,3,4,5,6,7,8,9 \
    scenario=ddpg \
    system=inv_pendulum \
    --experiment=ddpg_inv_pendulum \
    scenario.N_episodes=8 \
    scenario.N_iterations=100 \
    scenario.critic_opt_method_kwargs.lr=0.03 \
    scenario.critic_n_epochs=30 \
    scenario.policy_opt_method_kwargs.lr=0.03 \
    scenario.policy_n_epochs=1 \
    scenario.policy_model.n_hidden_layers=2 \
    scenario.policy_model.dim_hidden=15 \
    scenario.critic_model.n_hidden_layers=2 \
    scenario.critic_model.dim_hidden=80 \
    +scenario.critic_model.is_force_infinitesimal=True \
    scenario.discount_factor=0.75 
