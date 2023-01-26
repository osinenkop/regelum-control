python3 preset_endpoint.py \
--experiment=3wrobot \
controller=ddqn \
is_playback=False \
controller.critic.optimizer.opt_options.lr=0.1,0.001,0.000001 \
controller.critic.model.dim_hidden=20,100,500 \
controller.actor.epsilon_greedy_parameter=0.,0.15
