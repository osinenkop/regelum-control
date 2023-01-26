 python preset_endpoint.py \
     controller=calf \
     --experiment=3wrobot \
     is_playback=false \
     controller/critic/model=quadratic,quad_no_mix,quad_lin \
     controller.critic.critic_regularization_param=0.1,10,1000 \
     controller.critic.safe_decay_param=10,10000,100
