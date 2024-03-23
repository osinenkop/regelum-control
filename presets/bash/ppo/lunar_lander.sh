run_experiment_with_seed() {
    local seed="$1"
    python run_stable.py \
    +seed=$seed \
    --single-thread \
    scenario=ppo \
    system=lunar_lander \
    --experiment=ppo_lunar_lander \
    scenario.N_episodes=10 \
    --parallel \
    scenario.N_iterations=100 \
    scenario.policy_n_epochs=30 \
    scenario.critic_n_epochs=30 \
    scenario.policy_opt_method_kwargs.lr=0.0005 \
    scenario.policy_model.n_hidden_layers=2 \
    scenario.policy_model.dim_hidden=15 \
    scenario.policy_model.std=0.1 \
    scenario.critic_model.n_hidden_layers=2 \
    scenario.critic_model.dim_hidden=15 \
    scenario.critic_opt_method_kwargs.lr=0.1 \
    scenario.gae_lambda=0.96 \
    simulator.time_final=4
}

for seed in {1..10}; do
  run_experiment_with_seed "$seed"
done