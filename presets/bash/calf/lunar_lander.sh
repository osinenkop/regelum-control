python run_stable.py \
+seed=range\(10\) \
scenario=calf \
system=lunar_lander \
scenario.critic_model.quad_matrix_type=symmetric \
--experiment=calf_lunar_lander \
scenario.N_iterations=5 \
+scenario.critic_safe_decay_param=5 \
+scenario.critic_lb_parameter=1.0E-1 \
+scenario.critic_regularization_param=1000 \
+scenario.critic_learning_norm_threshold=3 \
scenario.critic_td_n=3 \
scenario.critic_batch_size=80 \
simulator.time_final=4 \
+scenario.critic_model.add_random_init_noise=True