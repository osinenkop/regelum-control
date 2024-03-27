python run_stable.py \
+seed=0,1,2,3,4,5,6,7,8,9 \
scenario=calf \
system=kin_point \
--experiment=calf_kin_point \
--jobs=-1 \
+scenario.critic_safe_decay_param=0.01 \
+scenario.critic_lb_parameter=1.0E-3 \
+scenario.critic_regularization_param=10000 \
+scenario.critic_learning_norm_threshold=0.01 \
scenario.critic_model.quad_matrix_type=symmetric \
scenario.N_iterations=6 \
+scenario.critic_model.add_random_init_noise=True