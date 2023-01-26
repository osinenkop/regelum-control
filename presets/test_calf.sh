 python preset_endpoint.py \
     controller=calf_ex_post \
     --experiment=3wrobot \
     is_playback=false \
     controller.critic.critic_regularization_param=10,1000,1000000 \
     controller.critic.safe_decay_param=100,10000
