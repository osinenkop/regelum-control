python run_stable.py \
+seed=1,4 \
--jobs=-1 \
scenario=rpv \
system=inv_pendulum \
--experiment=calfinit_inv_pendulum \
scenario.critic_model.quad_matrix_type=symmetric \
+scenario.critic_regularization_param=50000 \
scenario.N_iterations=1 \
+scenario.critic_model.add_random_init_noise=True