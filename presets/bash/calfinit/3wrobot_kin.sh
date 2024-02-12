python run_stable.py \
+seed=range\(10\) \
--jobs=-1 \
scenario=rpv \
system=3wrobot_kin \
--experiment=calfinit_3wrobot_kin \
scenario.critic_model.quad_matrix_type=symmetric \
+scenario.critic_regularization_param=3000 \
scenario.N_iterations=1 \
+scenario.critic_model.add_random_init_noise=True