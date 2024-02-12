python run_stable.py \
+seed=range\(10\) \
--jobs=-1 \
scenario=rpv \
system=lunar_lander \
--experiment=calfinit_lunar_lander \
scenario.critic_model.quad_matrix_type=symmetric \
+scenario.critic_regularization_param=1000 \
scenario.N_iterations=1 \
+scenario.critic_model.add_random_init_noise=True \
simulator.time_final=4