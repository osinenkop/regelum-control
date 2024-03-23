python run_stable.py \
    +seed=0,1,2,3,4,5,6,7,8,9 \
    --jobs=-1 \
    scenario=reinforce \
    system=cartpole_pg \
    --experiment=reinforce_cartpole \
    scenario.N_episodes=3 \
        scenario.N_iterations=500 \
    scenario.policy_opt_method_kwargs.lr=0.05 \
    scenario.policy_model.n_hidden_layers=2 \
    scenario.policy_model.dim_hidden=[32,32] \
    scenario.discount_factor=0.9 \
    scenario.policy_model.normalize_output_coef=0.05 \
    system_specific.time_final=15