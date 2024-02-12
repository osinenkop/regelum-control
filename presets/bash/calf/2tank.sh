python run_stable.py \
+seed=0,1,2,3,4,5,6,7,8,9 \
--jobs=-1 \
scenario=calf \
system=2tank \
--experiment=calf_2tank \
+scenario.critic_safe_decay_param=0.01 \
scenario.critic_model.quad_matrix_type=symmetric \
+scenario.critic_lb_parameter=1.0E-2 \
+scenario.critic_regularization_param=10000 \
+scenario.critic_learning_norm_threshold=0.05 \
+scenario.store_weights_thr=1. \
scenario.N_iterations=6 \
+scenario.critic_model.add_random_init_noise=True